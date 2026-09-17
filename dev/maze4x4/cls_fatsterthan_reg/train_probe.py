"""Compare frozen-teacher action classification and full Q-vector regression.

Run directly with Python. Each teacher supplies labels for BOTH objectives.
A step is one minibatch gradient update per student, not an environment step.
"""

import argparse
import copy
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from IncompleteBaseline.algorithms.common.qnetwork import QNetwork
from IncompleteBaseline.envs.procedual_maze.env import Maze

HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dqn-target', type=Path, default=HERE / 'models/q_target_network_dqn.pt')
    parser.add_argument('--actordqn-target', type=Path, default=HERE / 'models/q_target_network_actordqn.pt')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'probe_results')
    parser.add_argument('--num-transitions', type=int, default=102_400)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--steps', type=int, default=102_400)
    parser.add_argument('--eval-every', type=int, default=10_000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--max-episode-steps', type=int, default=1600)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    for name in ('num_transitions', 'steps', 'eval_every', 'batch_size', 'max_episode_steps'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if not 0 < args.validation_fraction < 1 or args.learning_rate <= 0:
        parser.error('Require 0 < validation-fraction < 1 and learning-rate > 0')
    return args


def sample_transitions(args):
    env = Maze(width=4, height=4, train_step_limitation=args.max_episode_steps)
    rng = np.random.default_rng(args.seed)
    rows = []
    try:
        state, _ = env.reset(seed=args.seed)
        for _ in range(args.num_transitions):
            action = int(rng.integers(env.action_space.n))
            next_state, reward, terminated, truncated, _ = env.step(action)
            rows.append((state.reshape(-1).copy(), action, reward,
                         next_state.reshape(-1).copy(), terminated, truncated))
            state = next_state
            if terminated or truncated:
                state, _ = env.reset()
    finally:
        env.close()
    names = ('states', 'actions', 'rewards', 'next_states', 'terminated', 'truncated')
    # like a pd.DataFrame
    return {name: torch.from_numpy(np.asarray(values))
            for name, values in zip(names, zip(*rows))}


def split_states(states, fraction, seed):
    # Preserve every transition, but keep identical input states in one split.
    unique, inverse = torch.unique(states, dim=0, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('Need at least two distinct states; sample more transitions.')
    order = torch.randperm(len(unique), generator=torch.Generator().manual_seed(seed))
    # if the size of fraction is larger than len(unique),
    # then count(the size of evaluation set) be set to len(unique)-1
    # means it at most only has len(unique)-1 unique transitions in the sampling set.
    count_eval = min(len(unique) - 1, max(1, int(len(unique) * fraction)))
    validation_groups = torch.zeros(len(unique), dtype=torch.bool)
    validation_groups[order[:count_eval]] = True
    # validation_groups: whether the transition is in the evaluation set
    # inverse: the index of the transition in the original set
    # mask: False means all the instances of that transition in the original set are not in the evaluation set
    mask = validation_groups[inverse]
    #           training set      evaluation set
    return torch.where(~mask)[0], torch.where(mask)[0]


@torch.no_grad()
def predict(model, states):
    return torch.cat([model(batch) for batch in states.split(4096)])


@torch.no_grad()
def evaluate(model, states, targets, labels, task):
    model.eval()
    correct, squared_error = 0, 0.0
    for start in range(0, len(states), 4096):
        output = model(states[start:start + 4096])
        if task == 'classification':
            actions = torch.multinomial(output.softmax(dim=1), 1).squeeze(1)
        else:
            actions = output.argmax(1)
        correct += (actions == labels[start:start + 4096]).sum().item()
        squared_error += (output - targets[start:start + 4096]).square().sum().item()
    model.train()
    return correct / len(states), squared_error / targets.numel()


def main():
    args = parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    teachers = {}
    # {
    #     'dqn': QNetwork(32, 4),
    #     'actordqn': QNetwork(32, 4),
    # }
    for name, path in [('dqn', args.dqn_target), ('actordqn', args.actordqn_target)]:
        teacher = QNetwork(32, 4)
        teacher.load_state_dict(torch.load(path.expanduser(), map_location='cpu', weights_only=True))
        teacher.eval().requires_grad_(False)
        teachers[name] = teacher

    print('Sampling random-policy transitions...', flush=True)
    transitions = sample_transitions(args)
    states = transitions['states'].float()
    train_idx, val_idx = split_states(states, args.validation_fraction, args.seed)
    print(f'Train: {len(train_idx)} transitions; validation: {len(val_idx)} transitions '
          '(no shared input states).', flush=True)
    # {
    #     'dqn': torch.Tensor num_transitions x 32
    #     'actordqn': torch.Tensor num_transitions x 32
    # }
    targets = {name: predict(teacher, states) for name, teacher in teachers.items()}
    # {
    #     'dqn': torch.Tensor num_transitions x 1
    #     'actordqn': torch.Tensor num_transitions x 1
    # }
    labels = {name: values.argmax(1) for name, values in targets.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'transitions': transitions, 'train_indices': train_idx,
                'validation_indices': val_idx, 'q_targets': targets, 'labels': labels,
                'config': {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()}}, args.output_dir / 'dataset.pt')

    # All four students start with exactly the same weights.
    torch.manual_seed(args.seed)
    initial = QNetwork(32, 4)
    # {
    #     (qdn, classification): QNetwork(32, 4),
    #     (qdn, regression): QNetwork(32, 4),
    #     (actor, classification): QNetwork(32, 4),
    #     (actor, regression): QNetwork(32, 4),
    # }
    students = {(teacher, task): copy.deepcopy(initial)
                for teacher in teachers for task in ('classification', 'regression')}
    optimizers = {key: torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                  for key, model in students.items()}
    losses = {'classification': nn.CrossEntropyLoss(), 'regression': nn.MSELoss()}
    generator = torch.Generator().manual_seed(args.seed + 1)
    val_states = states[val_idx]
    val_targets = {name: values[val_idx] for name, values in targets.items()}
    val_labels = {name: values[val_idx] for name, values in labels.items()}

    with SummaryWriter(str(args.output_dir / 'tensorboard')) as writer, \
            (args.output_dir / 'metrics.csv').open('w', newline='') as file:
        metrics = csv.writer(file)
        metrics.writerow(['step', 'samples_seen', 'teacher', 'task', 'action_accuracy', 'q_mse'])
        for step in range(args.steps + 1):
            if step > 0:
                # Same minibatch for both objectives and both teachers.
                indices = train_idx[torch.randint(len(train_idx), (args.batch_size,), generator=generator)]
                batch = states[indices]
                for key, student in students.items():
                    teacher, task = key
                    target = labels[teacher][indices] if task == 'classification' else targets[teacher][indices]
                    loss = losses[task](student(batch), target)
                    optimizers[key].zero_grad()
                    loss.backward()
                    optimizers[key].step()
            if step % args.eval_every == 0 or step == args.steps:
                # students:
                # {
                #     (qdn, classification): QNetwork(32, 4),
                #     (qdn, regression): QNetwork(32, 4),
                #     (actor, classification): QNetwork(32, 4),
                #     (actor, regression): QNetwork(32, 4),
                # }
                for (teacher, task), student in students.items():
                    accuracy, mse = evaluate(student, val_states, val_targets[teacher], val_labels[teacher], task)
                    # Classification logits have no Q-value scale; do not report their MSE.
                    q_mse = mse if task == 'regression' else ''
                    metrics.writerow([step, step * args.batch_size, teacher, task, accuracy, q_mse])
                    writer.add_scalar(f'{teacher}/{task}/action_accuracy', accuracy, step)
                    if task == 'regression':
                        writer.add_scalar(f'{teacher}/{task}/q_mse', mse, step)
                    print(f'step={step:6d} teacher={teacher:8s} {task:14s} '
                          f'action_accuracy={accuracy:.4f}', flush=True)
                file.flush()
                writer.flush()
    print(f'Results: {args.output_dir}')


if __name__ == '__main__':
    main()
