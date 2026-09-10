# 确认类是否可放入集合

> 我想提高在 random rollout 的时候的采样的多样性，现在正在实现。
>
> 我有个问题：它这个类能够被放到一个集合里面吗？
>
> User context
>
> Images:
> - `/var/folders/kw/d1s4p1k565q1hwqndgb055bh0000gn/T/codex-clipboard-e8448104-e1d5-4686-b4f4-d928a99aaf3b.png`
> - `/var/folders/kw/d1s4p1k565q1hwqndgb055bh0000gn/T/codex-clipboard-0d240458-9cdb-41e2-a752-0f4b697bc927.png`
> - `/var/folders/kw/d1s4p1k565q1hwqndgb055bh0000gn/T/codex-clipboard-5a4d1b78-676d-45ad-8022-a300a39dcd1d.png`

<details><summary>2 previous messages</summary>

> 我先对照你截图里的 `MABObject` 和已有的状态去重记录，确认放进 `set` 后会怎样判断重复。
>
> <details><summary>Explored a search</summary>
>
> - Searched for `Tensor object|对象|immutable|state counting|470|166` in `MEMORY.md`
>
> </details>
</details>

**可以，截图中 `MABObject` 的实例可以放进 `set`，但默认不会按 `state` 的内容去重。**

你没有定义 `__eq__` 和 `__hash__`，所以 Python 默认按**对象身份**判断是否重复：

```python
a = MABObject(state)
b = MABObject(state)

buffer = {a, b}
print(len(buffer))  # 2，即使传入的是同一个 state

buffer.add(a)
print(len(buffer))  # 2，重复添加同一个对象才会去重
```

`self.actions` 是可变的列表，也不影响这个默认行为。另外，你当前的 `__init__` 接收了 `state`，但还没有保存它。

按照你的需求——**每个状态维护一个 MAB，再次遇到时取出来更新**——用 `dict` 更直接：

```python
class MABRollout:
    def __init__(self, env: gym.Env):
        self.env = env
        self.buffer = {}

    def step(self, state: torch.Tensor):
        # 假设 observation 的形状固定，按 Tensor 的数值识别状态
        key = tuple(state.detach().cpu().reshape(-1).tolist())

        if key not in self.buffer:
            self.buffer[key] = MABObject(state)

        mab = self.buffer[key]
        # 接下来使用 mab.actions 选择动作并更新
```

这里 `key` 负责识别状态，`mab` 保存该状态下不断更新的动作信息。**不要直接把 Tensor 当作去重的 key**，它不会按你需要的数值内容识别相同状态。

> 它的形状不固定会怎么样？

如果形状不固定，**展平之后可能把不同形状的 Tensor 当成同一个状态**：

```python
a = torch.tensor([[1, 2], [3, 4]])  # shape: (2, 2)
b = torch.tensor([1, 2, 3, 4])      # shape: (4,)

# 展平后的 key 都是 (1, 2, 3, 4)
```

如果形状也是状态的一部分，就把 **shape 和数值一起放进 key**：

```python
key = (
    tuple(state.shape),
    tuple(state.detach().cpu().reshape(-1).tolist()),
)
```

此时分别得到：

```python
# a
((2, 2), (1, 2, 3, 4))

# b
((4,), (1, 2, 3, 4))
```

这样形状或数值不同，都会被视为不同状态。不过，如果 `(4,)` 和 `(1, 4)` 只是同一状态有没有 batch 维度的区别，应先统一表示，避免重复建 MAB。

> 其实是不是可以直接在这个 MAB Rollout 里面实现？它的 key 就是那个 state，然后 value 就是它的那个 action 的一个 preference

对，**可以直接在 `MABRollout` 里用字典维护「状态 → 各动作的 preference」**，这样暂时不需要单独的 `MABObject` 类。

只是字典的 key 应该用 `state` 转换后的不可变表示，而不是直接使用 Tensor：

```python
class MABRollout:
    def __init__(self, env: gym.Env):
        self.env = env
        self.preferences = {}

    def get_preferences(self, state: torch.Tensor):
        key = (
            tuple(state.shape),
            tuple(state.detach().cpu().reshape(-1).tolist()),
        )

        if key not in self.preferences:
            self.preferences[key] = [
                math.inf for _ in range(self.env.action_space.n)
            ]

        return self.preferences[key]
```

使用时：

```python
preferences = self.get_preferences(state)

# 更新当前 state 对应的某个动作
preferences[action] = 1.0
```

**返回的列表就是字典中保存的那个列表**，所以修改 `preferences[action]` 会直接更新字典中的值；下次遇到相同状态，就会取到更新后的 preference。

> 为什么不能直接用 Tensor？

**Tensor 可以作为字典的 key，但默认按“对象身份”区分，不会按数值内容匹配。**

例如，下面两个 Tensor 数值相同，却是两个独立对象：

```python
a = torch.tensor([1, 2])
b = torch.tensor([1, 2])

buffer = {a: [float("inf")] * 4}

print(a in buffer)  # True
print(b in buffer)  # False
```

这对你的 rollout 意味着：**再次遇到相同状态时，如果环境返回了一个新的 Tensor，就查不到之前保存的 preference。**

转换成 tuple 后，就会按内容匹配：

```python
buffer = {tuple(a.tolist()): [float("inf")] * 4}

print(tuple(b.tolist()) in buffer)  # True
```

还有反过来的问题：如果环境一直复用同一个 Tensor，并原地修改其数值，那么状态已经变了，字典仍然会认作同一个 key。

所以你需要的是**状态内容的不可变快照**，而 Tensor 本身不提供这种字典键语义。

> 帮我 check 一下我的实现，对不对？

<details><summary>2 previous messages</summary>

> 我会检查你现在的 `MABRollout` 实现，以及它和环境、rollout 调用的衔接，先只 review，不修改文件。
>
> <details><summary>Explored a search, ran 3 commands</summary>
>
> - Ran `pwd; rg --files -g AGENTS.md -g '*rollout*' -g '*agent.py' -g '*maze*' project/IncompleteBaseline`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `Tensor-value|flattened immutable` in `MEMORY.md`
>
> </details>
>
> - Ran `for p in /AGENTS.md /Users/AGENTS.md /Users/jay/AGENTS.md /Users/jay/Desktop/AGENTS.md /Users/jay/Desktop/RL/AGENTS.md /Users/jay/Desktop/RL/project/AGENTS.md; do if [ -f "$p" ]; then cat "$p"; fi; done
> rg --files --hidden project/IncompleteBaseline -g AGENTS.md
> nl -ba project/IncompleteBaseline/algorithms/common/rollout.py
> rg -n 'MABRollout|MABObject|random_rollout|rollout|reward|observation' project/IncompleteBaseline/algorithms/dqn/dqn_agent.py project/IncompleteBaseline/envs -g '*.py'`
>
> - Ran `rg -n 'MABRollout|TensorObservation|common.rollout' project/IncompleteBaseline project/ActorDQN -g '*.py'
> sed -n '245,270p' project/IncompleteBaseline/algorithms/dqn/dqn_agent.py
> /Users/jay/miniconda3/envs/rl/bin/python -B - <<'PY'
> import sys, torch, gymnasium as gym
> sys.path.insert(0, '/Users/jay/Desktop/RL/project/IncompleteBaseline')
> from algorithms.common.rollout import MABRollout
> class Env:
>     action_space = gym.spaces.Discrete(4)
>     def step(self, action):
>         return torch.tensor([2.]), 0., False, False, {}
> m = MABRollout(Env(), c=1.)
> s = torch.tensor([1.])
> for label, fn in [('first visit', lambda: m.step(s, 1)), ('second visit', lambda: m.step(s, 2)), ('float call', lambda: m.lr(1.)), ('inf update', lambda: torch.tensor(float('inf')) + m.lr * (1 - torch.tensor(float('inf'))))]:
>     try: print(label, fn())
>     except Exception as e: print(label, type(e).__name__, str(e))
> PY`
>
> </details>
</details>

看了当前的 [rollout.py](./project/IncompleteBaseline/algorithms/common/rollout.py)，**字典按状态保存信息的方向对，但目前还不能正常运行，更新逻辑也有几处问题。**没有修改文件。

1. **使用了未定义的属性**（[第 41 行](./project/IncompleteBaseline/algorithms/common/rollout.py:41)）

   `self.output_dim` 和 `self.training_env` 没有定义。这里应分别使用：
   ```python
   self.env.action_space.n
   self.env.step(action)
   ```

2. **UCB 公式取错了变量**（[第 45 行](./project/IncompleteBaseline/algorithms/common/rollout.py:45)）

   `p_info` 是 `[Q, N]` 列表，不能直接加 Tensor。应该取出：
   ```python
   q, n = self.preferences[key]
   p = q + self.c * torch.sqrt(math.log(time_step) / n)
   ```
   `time_step` 必须大于零。如果每个状态是独立的 bandit，更合理的是使用**该状态的访问次数**作为这里的时间。

3. **更新写到了计数 N 上，而且少了乘号**（[第 58 行](./project/IncompleteBaseline/algorithms/common/rollout.py:58)）

   `[1]` 是计数，`[0]` 才是 Q；`self.lr(...)` 则是在把浮点数当函数调用。应当是：
   ```python
   q, n = self.preferences[key]
   n[action] += 1
   q[action] += self.lr * (r - q[action])
   ```

4. **Q 初始化为 `inf`，无法这样做增量更新**（[第 38 行](./project/IncompleteBaseline/algorithms/common/rollout.py:38)）

   即便修正上一条，也会得到：
   ```python
   inf + lr * (r - inf)  # nan
   ```
   建议 **Q 和 N 初始化为零，优先随机选择 `N == 0` 的动作；全部尝试过后再使用 UCB**。这样也不需要用 `1e-8` 假装已经采样过。

5. **首次遇到状态时，没有记录这次动作的反馈**（[第 36 行](./project/IncompleteBaseline/algorithms/common/rollout.py:36)）

   新状态分支只执行环境动作，计数和 Q 更新都在 `else` 内。应把流程统一成：

   ```text
   必要时初始化 → 选择动作 → env.step → 计算探索奖励 → 更新 N 和 Q
   ```

6. **两个方法保存的数据结构不一致**（[第 16 行](./project/IncompleteBaseline/algorithms/common/rollout.py:16)）

   `get_preferences()` 保存的是动作 preference 列表，`step()` 保存的是 `[Q_tensor, N_tensor]`。如果先调用前者，后者就会读到错误结构。应统一初始化入口。

另外，你现在的奖励判断是 **“下一状态以前是否访问过”**，而图里写的是 **“动作后是否停留在原状态”**。这两种设计不同：当前实现会惩罚回到任何已访问状态，包括迷宫里必要的回退。需要明确你想实现哪一种。

我用最小环境复现了未定义属性、列表加 Tensor、浮点数调用和 `inf` 更新产生 `nan` 的问题。目前也没有找到 `MABRollout` 的调用处，所以它尚未影响现有的 random rollout。

> 这里改的是 N，它也会把 preferences 里面的那个东西也改了吗？

:codex-annotation{index="1"} 会的。这里的解包**不会复制 Tensor**：

```python
q, n = self.preferences[key]
```

`n` 和 `self.preferences[key][1]` 指向同一个 Tensor，所以：

```python
n[action] += 1
```

会原地修改它，`self.preferences` 里的计数也就一起更新了。

注意下面两种写法的区别：

```python
n += 1      # 原地修改，字典中的 Tensor 也会变
n = n + 1   # 创建新 Tensor，再让 n 指向它；字典中的 Tensor 不变
```

如果希望得到独立副本，需要显式复制：

```python
n = self.preferences[key][1].clone()
```

> 现在再看一下，上面 6 个问题有没有解决？

<details><summary>2 previous messages</summary>

> 我会重新读取当前文件，逐项核对上面 6 个问题。
>
> <details><summary>Explored a file, ran a command</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./project/IncompleteBaseline/algorithms/common/rollout.py`
>
> </details>
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B - <<'PY'
> import sys, torch, gymnasium as gym, math
> sys.path.insert(0, '/Users/jay/Desktop/RL/project/IncompleteBaseline')
> from algorithms.common.rollout import MABRollout
> class Env:
>     action_space = gym.spaces.Discrete(4)
>     def step(self, action):
>         return torch.tensor([2.]), -0.01, False, False, {}
> s = torch.tensor([1.])
> m = MABRollout(Env(), c=1.)
> try: m.step(s, 1)
> except Exception as e: print('first step:', type(e).__name__, str(e))
> print('Q dtype:', m.preferences[(1.,)][0].dtype)
> for t in [0, 1, 2]:
>     try: print('UCB t=', t, torch.sqrt(math.log(t)/torch.tensor([1, 0, 0, 0])))
>     except Exception as e: print('UCB t=', t, type(e).__name__, str(e))
> m2 = MABRollout(Env(), c=1.)
> m2.get_preferences(s)
> try: m2.step(s, 2)
> except Exception as e: print('get then step:', type(e).__name__, str(e))
> PY`
>
> </details>
</details>

重新检查了当前的 [rollout.py](./project/IncompleteBaseline/algorithms/common/rollout.py)。**第 1、3、5 项已修正；第 2、4 项还不完整；第 6 项未解决。另外新增了一个 Q 的类型问题。**

| 原问题 | 当前情况 |
|---|---|
| 1. 未定义的属性 | ✅ 已改为 `self.env.action_space.n` 和 `self.env.step()` |
| 2. UCB 取错变量 | ⚠️ 已正确使用 `q`，但仍未处理零计数和 `time_step` 边界 |
| 3. 更新错索引、缺乘号 | ✅ 已正确更新 `n[action]` 和 `q[action]` |
| 4. Q 初始化为 `inf` | ⚠️ 已改成零，但没有显式优先选择未尝试动作 |
| 5. 首次访问不更新 | ✅ 新旧状态现在共用反馈更新 |
| 6. 两种存储结构不一致 | ❌ `get_preferences()` 仍然保存浮点列表，`step()` 保存 `[Q, N]` |

还需要改的具体地方：

**① Q 必须初始化为浮点 Tensor。**
[第 38 行](./project/IncompleteBaseline/algorithms/common/rollout.py:38) 的整数列表会产生 `torch.int64`，第 59 行更新时直接报错。我已复现：

```text
RuntimeError: result type Float can't be cast to the desired output type Long
```

改为：

```python
torch.zeros(self.env.action_space.n, dtype=torch.float32),  # Q
torch.zeros(self.env.action_space.n, dtype=torch.int64),    # N
```

**② 先处理未尝试的动作，再计算 UCB。**
当前公式在 `time_step=0` 时报错；在 `time_step=1` 且 `n=0` 时产生 `nan`。可以把选动作统一写成：

```python
q, n = self.preferences[key]
untried = torch.where(n == 0)[0].tolist()

if untried:
    action = self._rng.choice(untried)
else:
    # 每个状态作为独立的 bandit，使用该状态的累计动作次数
    t = n.sum().item()
    p = q + self.c * torch.sqrt(math.log(t) / n)
    action = torch.argmax(p).item()
```

**③ 统一 `get_preferences()` 的初始化结构，或者删除这个未使用的方法。**
现在先调用它再调用 `step()`，我复现了：

```text
ValueError: too many values to unpack (expected 2)
```

以上只做了检查和最小复现，没有修改文件。

> 再确认一下

<details><summary>2 previous messages</summary>

> 我再读取一下最新文件，确认剩余问题是否已修正。
>
> <details><summary>Explored a file, ran a command</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./project/IncompleteBaseline/algorithms/common/rollout.py`
>
> </details>
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B - <<'PY'
> import sys, torch, gymnasium as gym
> sys.path.insert(0, '/Users/jay/Desktop/RL/project/IncompleteBaseline')
> from algorithms.common.rollout import MABRollout
> class Env:
>     action_space = gym.spaces.Discrete(4)
>     def __init__(self): self.actions=[]
>     def step(self, action):
>         self.actions.append(action)
>         return torch.tensor([1., 2.]), -0.05, False, False, {}
> e=Env(); m=MABRollout(e,c=1.)
> for i in range(20):
>     m.step(torch.tensor([1., 2.]))
>     q,n=m.preferences[(1.,2.)]
>     assert torch.isfinite(q).all()
>     assert n.sum().item()==i+1
> assert len(set(e.actions[:4]))==4
> assert len(m.preferences)==1
> print('PASS: 20 steps; first four actions',e.actions[:4],'; N',n.tolist(),'; Q',q.tolist())
> class NewEnv:
>     action_space=gym.spaces.Discrete(4)
>     def step(self, action): return torch.tensor([3.,4.]), -0.01, False, False, {}
> m=MABRollout(NewEnv(),c=1.); m.step(torch.tensor([1.,2.])); q,n=m.preferences[(1.,2.)]
> assert torch.isclose(q[n>0],torch.tensor([0.099])).all()
> print('PASS: new-state reward updates selected Q to 0.099')
> PY`
>
> </details>
</details>

确认了，当前 [rollout.py](./project/IncompleteBaseline/algorithms/common/rollout.py) **之前的 6 个问题，以及后来发现的 Q 类型问题，都已解决。**

我用最小环境实际验证了：

- 连续运行 20 步没有异常，Q 没有出现 `nan`。
- 前 4 次分别选择了 4 个不同动作，之后进入 UCB。
- 每一步都正确更新字典中的 N 和 Q。
- 数值相同的新 Tensor 会复用同一份记录。
- 首次转移到新状态时，探索奖励正确参与 Q 更新。

还需保留两个语义上的注意点：

- 当前奖励区分的是**下一状态是否访问过**，因此回到已访问状态也会受罚，并非只惩罚原地不动。
- key 仍未包含形状；如果不同形状代表不同状态，需要把 `tuple(state.shape)` 加进去。

**这次确认的是该类的局部运行与更新逻辑；是否实际提高采样多样性，还需要接入 rollout 后测量。**未修改文件。

> 帮我 check 一下 DQN agent 那个文件。我现在把它改成了：
>
> 1. 在随机 rollout 的时候有一个随机 rollout 策略：如果写的是 MAB 就用 MAB；没写（None / 为空）的话就走 uniform 随机采样。
> 2. 还有一个开关，就是 epsilon-greedy 的 exploration：当采样到的概率小于 10%（exploration rate）的时候，也分为采用 MAB 或者随机采样。
>
> 你看一下我现在写得对不对？

<details><summary>2 previous messages</summary>

> 我会检查 DQN agent 里这两处策略分支，以及 MAB 调用和 replay buffer 的衔接，确认是否符合你描述的行为。
>
> <details><summary>Explored 2 files, a search, a list, ran a command</summary>
>
> <details><summary>Explored 2 files, a search, a list</summary>
>
> - Searched for `DQN|rollout` in `MEMORY.md`
> - Read `./project/IncompleteBaseline/algorithms/dqn/dqn_agent.py`
> - Read `./project/IncompleteBaseline/algorithms/common/rollout.py`
> - Listed files in `project`
>
> </details>
>
> - Ran `sed -n '1,100p' project/IncompleteBaseline/algorithms/common/exploration_rate_calculation.py
> /Users/jay/miniconda3/envs/rl/bin/python -B - <<'PY'
> import sys, torch, gymnasium as gym
> from unittest.mock import patch
> sys.path.insert(0, '/Users/jay/Desktop/RL/project/IncompleteBaseline')
> from algorithms.dqn.dqn_agent import DQNAgent
> class Env:
>     action_space=gym.spaces.Discrete(4)
>     def __init__(self): self.calls=0
>     def step(self,a):
>         self.calls+=1
>         return torch.tensor([[0.,1.]]), -.01, False, False, {}
> class Eps:
>     def step(self,t): return 1.
> with patch('algorithms.dqn.dqn_agent.SummaryWriter'):
>     for warm,eps in [(None,None),('mab',None),(None,'mab'),('mab','mab'),('',None),(None,''),('MAB',None)]:
>         env=Env()
>         try:
>             a=DQNAgent(2,4,7,20,env,Env(),2,epsilon_strategy=Eps(),total_time_steps=20,learning_start=5,eval_freq=None,random_rollout_stategy=warm,eps_exp_strategy=eps)
>             for name,call in [('warmup',lambda:a.random_rollout(torch.tensor([[0.,1.]]))),('explore',lambda:a.rollout(1,torch.tensor([[0.,1.]])))]:
>                 before=env.calls
>                 try:
>                     call(); print(repr((warm,eps)),name,'PASS', 'env steps',env.calls-before,'buffer',len(a.replaybuffer.pool))
>                 except Exception as e: print(repr((warm,eps)),name,type(e).__name__,str(e))
>         except Exception as e: print(repr((warm,eps)),'init',type(e).__name__,str(e))
> PY`
>
> </details>
</details>

**整体分支思路正确，但有一个调用错误，导致两个开关还不能独立使用；空字符串也没有处理。**我用最小环境测试了各组合，没有修改文件。

| warmup 策略 | ε-exploration 策略 | 实测结果 |
|---|---|---|
| `None` | `None` | ✅ 两处均 uniform |
| `"mab"` | `None` | ✅ warmup 用 MAB，探索用 uniform |
| `None` | `"mab"` | ❌ 探索时报错 |
| `"mab"` | `"mab"` | ✅ 两处共用同一个 MAB，保留 warmup 的记录 |

**1. ε-exploration 调错了对象**

[第 251 行](./project/IncompleteBaseline/algorithms/dqn/dqn_agent.py:251) 判断的是 `self.eps_exp_strategy`，但调用了 `self.rollout_warmup.step()`。

应该改为：

```python
action, next_state, reward, terminated, truncated, info = (
    self.eps_exp_strategy.step(state=state)
)
```

否则 warmup 为 `None`、探索为 `"mab"` 时，会出现：

```text
AttributeError: 'NoneType' object has no attribute 'step'
```

**2. 默认不传和 `None` 可以；`""`、`"MAB"` 目前不可以**

[初始化分支](./project/IncompleteBaseline/algorithms/dqn/dqn_agent.py:100) 只处理精确的 `"mab"` 和 `None`。传空字符串或大写 `"MAB"` 会导致属性根本没有初始化。

如果你希望“空”也包含空字符串，可以先统一处理两个参数：

```python
random_rollout_stategy = (
    random_rollout_stategy.strip().lower() or None
    if random_rollout_stategy is not None else None
)
eps_exp_strategy = (
    eps_exp_strategy.strip().lower() or None
    if eps_exp_strategy is not None else None
)
```

然后对非 `None`、非 `"mab"` 的值抛出 `ValueError`，避免拼写错误被漏掉。

**3. 参数类型标注应与实际传入值一致**

[第 42 行](./project/IncompleteBaseline/algorithms/dqn/dqn_agent.py:42) 标的是 `MABRollout` 对象，实际判断的是字符串，应改为：

```python
random_rollout_stategy: str | None = None,
eps_exp_strategy: str | None = None,
```

其余衔接是对的：默认 `ClassicalExploration` 返回 `0.1`，`random() < exploration_rate` 进入探索，否则走 Q-network 贪心动作。通过的分支每次只调用一次环境，并向 replay buffer 写入一条 transition；保存的也是环境原始 reward。

另外，两处新建 `MABRollout` 时没有传 `seed=self.seed`，所以 MAB 始终使用默认种子 `42`。如果希望 agent 的 seed 同时控制 MAB，应补上传参。

> 帮我 check 一下 main.py，看一下我这个参数写的对不对？

<details><summary>2 previous messages</summary>

> 我会对照 `main.py` 的参数定义、传参位置和 DQNAgent 的接口检查。
>
> <details><summary>Explored a file, a search, ran a command</summary>
>
> <details><summary>Explored a file, a search</summary>
>
> - Searched for `random_rollout|eps_exp|mab|MAB|DQNAgent|ArgumentParser` in `main.py`
> - Read `./project/IncompleteBaseline/dev/main.py`
>
> </details>
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B - <<'PY'
> import ast,argparse,contextlib,io
> from typing import Sequence
> p='project/IncompleteBaseline/dev/main.py'
> t=ast.parse(open(p).read()); f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
> ns={'argparse':argparse,'Sequence':Sequence}; exec(compile(ast.Module(body=[f],type_ignores=[]),p,'exec'),ns)
> for args in [[],['--random-rollout-stategy','MABRollout'],['--random-rollout-stategy','mab'],['--epsilon-exploration-strategy','None']]:
>     try:
>         with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): r=ns['parse_args'](args)
>         print(args,'=>',repr(r.random_rollout_stategy),repr(r.epsilon_exploration_strategy))
>     except SystemExit as e: print(args,'=> parse rejected, code',e.code)
> PY`
>
> </details>
</details>

看了你新增参数的 [dev/main.py](./project/IncompleteBaseline/dev/main.py:58)。**传给 DQNAgent 的参数名对应正确，但参数值不一致，当前无法正常启用 MAB。**

你现在写的是：

```python
choices=["MABRollout", None]
```

但 DQNAgent 判断的是：

```python
if random_rollout_stategy == "mab":
```

所以 `"MABRollout"` 能通过命令行解析，却不会进入 agent 的 MAB 分支；`"mab"` 则被命令行解析器拒绝。我已实际验证。

建议把两处参数改成：

```python
parser.add_argument(
    "--epsilon-exploration-strategy",
    type=str,
    choices=["mab"],
    default=None,
    help="Exploration strategy; omit for uniform random sampling.",
)
parser.add_argument(
    "--random-rollout-stategy",
    type=str,
    choices=["mab"],
    default=None,
    help="Warmup strategy; omit for uniform random sampling.",
)
```

这样：

- **不传参数** → Python 的 `None` → uniform。
- **传 `--random-rollout-stategy mab`** → warmup 使用 MAB。
- **传 `--epsilon-exploration-strategy mab`** → ε-exploration 使用 MAB。

`default=None` 不要求把 `None` 放进 `choices`。命令行写 `None` 得到的是字符串 `"None"`，并不是 Python 的 `None`。

另外，`stategy` 是拼写错误，但你目前命令行、`args` 和 agent 三处一致，所以不会因此报错；如果改成 `strategy`，需要同步修改。没有修改文件。

> 现在再 check 一下两个文件

<details><summary>2 previous messages</summary>

> 我会重新核对 `main.py` 和 `dqn_agent.py`，并检查参数解析到策略分支的衔接。
>
> <details><summary>Explored 2 files, ran a command</summary>
>
> <details><summary>Explored 5 files</summary>
>
> - Read `./project/IncompleteBaseline/dev/main.py`
> - Read `./project/IncompleteBaseline/dev/main.py`
> - Read `./project/IncompleteBaseline/algorithms/dqn/dqn_agent.py`
> - Read `./project/IncompleteBaseline/algorithms/dqn/dqn_agent.py`
> - Read `./project/IncompleteBaseline/algorithms/dqn/dqn_agent.py`
>
> </details>
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B - <<'PY'
> import sys,ast,argparse,contextlib,io,itertools,torch,gymnasium as gym
> from typing import Sequence
> from unittest.mock import patch
> sys.path.insert(0,'/Users/jay/Desktop/RL/project')
> from IncompleteBaseline.algorithms.dqn.dqn_agent import DQNAgent
> p='project/IncompleteBaseline/dev/main.py'
> t=ast.parse(open(p).read()); f=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args'); ns={'argparse':argparse,'Sequence':Sequence}; exec(compile(ast.Module(body=[f],type_ignores=[]),p,'exec'),ns)
> class Env:
>     action_space=gym.spaces.Discrete(4)
>     def __init__(self): self.calls=0
>     def step(self,a):
>         self.calls+=1
>         return torch.tensor([[0.,1.]]),-.01,False,False,{}
> class Eps:
>     value=1.
>     def step(self,t): return self.value
> with patch('IncompleteBaseline.algorithms.dqn.dqn_agent.SummaryWriter'):
>     for w,e in itertools.product([None,'MABRollout'],repeat=2):
>         argv=[]
>         if w: argv+=['--random-rollout-stategy',w]
>         if e: argv+=['--epsilon-exploration-strategy',e]
>         with contextlib.redirect_stdout(io.StringIO()): args=ns['parse_args'](argv)
>         env=Env(); eps=Eps()
>         a=DQNAgent(2,4,7,20,env,Env(),2,epsilon_strategy=eps,total_time_steps=20,learning_start=5,eval_freq=None,random_rollout_stategy=args.random_rollout_stategy,eps_exp_strategy=args.epsilon_exploration_strategy)
>         s=torch.tensor([[0.,1.]])
>         a.random_rollout(s); a.rollout(1,s); eps.value=0.; a.rollout(2,s)
>         assert env.calls==3 and len(a.replaybuffer.pool)==3
>         if w and e: assert a.rollout_warmup is a.eps_exp_strategy
>         print(w,e,'PASS: warmup, exploration, greedy; 3 env steps and transitions')
> PY`
>
> </details>
</details>

**现在两个文件的策略参数已经对齐，核心逻辑正确。**检查了 [main.py](./project/IncompleteBaseline/dev/main.py:58) 和 [dqn_agent.py](./project/IncompleteBaseline/algorithms/dqn/dqn_agent.py:99)，并实际测试了参数解析及四种组合：

| warmup | ε-exploration | 结果 |
|---|---|---|
| 不传 | 不传 | ✅ 都用 uniform |
| `MABRollout` | 不传 | ✅ warmup 用 MAB |
| 不传 | `MABRollout` | ✅ 探索用 MAB |
| `MABRollout` | `MABRollout` | ✅ 共用同一个 MAB，继承 warmup 记录 |

之前调用错对象的问题已修正，MAB 也已传入 agent 的 seed。各组合的 warmup、探索、贪心分支均能执行，每次只推进环境一步，并保存一条 transition。

剩下两个小地方：

- `eps_exp_strategy: str = None` 建议改为 `str | None = None`。
- `choices=["MABRollout", None]` 中的 `None` 可以删掉，保留 `default=None`。**uniform 的用法是省略参数**，不是在命令行写 `None` 或空字符串。

没有修改文件；这次验证了策略分支，没有运行完整训练。

> 我在修改前和修改后发现，他们的训练结果没法复现，就是和我之前的训练结果不一样。
>
> 帮我看一下是哪里出问题了？都是随机种子 42 的。
> 就是我把它模块化之前和模块化之后，训练结果不一样。
> python -m IncompleteBaseline.dev.main --tensorboard-log-dir "IncompleteBaseline/dev/logs/qdn/maze5x5/dynamic\_epsilon/mabrollout/v1" \\
> &#x20;   \--width 5 \\
> &#x20;   \--height 5 \\
> &#x20;   \--max-episode-steps-eval 25 \\
> &#x20;   \--max-episode-steps 2500 \\
> &#x20;   \--max-episode-steps-warmup 5000 \\
> &#x20;   \--total-time-steps 1228800 \\
> &#x20;   \--learning-starts 819200 \\
> &#x20;   \--epsilon-strategy "StepDecay" \\
> &#x20;   \--epsilon-decay 204800 \\
> &#x20;   \--epsilon-exploration-strategy "MABRollout" \\
> &#x20;   \--random-rollout-stategy "MABRollout" \\
> &#x20;   \--seed 42
>
> User context
>
> Images:
> - `/var/folders/kw/d1s4p1k565q1hwqndgb055bh0000gn/T/codex-clipboard-a0cbc7f9-7666-4a35-a4f0-c503dc0d324f.png`

<details><summary>4 previous messages</summary>

> 我会先对比模块化前后的代码和两次运行的原始 TensorBoard 数据，找出最早出现差异的位置。相同 seed 还需要随机调用顺序和训练配置一致才能复现。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `rg -n 'raw scalar|EventAccumulator|reproducibility' /Users/jay/.codex/memories/MEMORY.md | head -15
>  git -C project/IncompleteBaseline status --short
>  git -C project/IncompleteBaseline log -6 --oneline
>  rg --files project/IncompleteBaseline | rg '(rollout|dqn_agent|main.py|events.out)'`
> - Ran `git -C project/IncompleteBaseline diff -- algorithms/common/rollout.py algorithms/dqn/dqn_agent.py dev/main.py dev/maze5x5/run_dqn_dyn.sh
> cat project/IncompleteBaseline/algorithms/common/rollout.py`
>
> </details>
>
> 代码里已经找到一处行为变化：旧版在 ε-exploration 时只用 MAB 选动作，不更新 MAB；新版调用 `MABRollout.step()`，会继续更新计数和偏好。我会用短对照验证它是否会改变动作序列，同时核对日志中的首次差异。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B - <<'PY'
> import sys,subprocess,types,torch,gymnasium as gym
> from unittest.mock import patch
> sys.path.insert(0,'/Users/jay/Desktop/RL/project')
> import IncompleteBaseline.algorithms.dqn.dqn_agent as cur
> repo='project/IncompleteBaseline'
> def oldmodule(path,name,package):
>  m=types.ModuleType(name); m.__package__=package
>  exec(compile(subprocess.check_output(['git','-C',repo,'show','HEAD:'+path],text=True),name,'exec'),m.__dict__); return m
> oldmab=oldmodule('algorithms/common/rollout.py','oldmab','IncompleteBaseline.algorithms.common')
> oldagent=oldmodule('algorithms/dqn/dqn_agent.py','oldagent','IncompleteBaseline.algorithms.dqn'); oldagent.MABRollout=oldmab.MABRollout
> class Env:
>  action_space=gym.spaces.Discrete(4)
>  def __init__(self):self.actions=[]
>  def step(self,a):self.actions.append(a);return torch.tensor([[0.,1.]]),-.05,False,False,{}
> class Eps:
>  def step(self,t):return 1.
> class Writer:
>  def __init__(self,*a,**k):pass
> cur.SummaryWriter=Writer;oldagent.SummaryWriter=Writer
> agents=[]
> for module,extra in [(oldagent,{}),(cur,{'random_rollout_stategy':'MABRollout','eps_exp_strategy':'MABRollout'})]:
>  torch.manual_seed(42); env=Env(); a=module.DQNAgent(2,4,42,100,env,Env(),2,epsilon_strategy=Eps(),total_time_steps=100,learning_start=10,eval_freq=None,**extra); agents.append(a)
>  for _ in range(8):a.random_rollout(torch.tensor([[0.,1.]]))
>  print(module.__name__,'warmup actions',env.actions,'N',a.rollout_warmup.preferences[(0.,1.)][1].tolist())
>  for t in range(1,6):a.rollout(t,torch.tensor([[0.,1.]]))
>  print('explore actions',env.actions[8:],'N',a.rollout_warmup.preferences[(0.,1.)][1].tolist())
> assert all(torch.equal(x,y) for x,y in zip(agents[0].qnetwork.parameters(),agents[1].qnetwork.parameters()))
> print('Initial network weights identical; no optimizer updates in comparison')
> PY`
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B - <<'PY'
> from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
> from pathlib import Path
> base=Path('project/IncompleteBaseline/dev/logs/qdn/maze5x5/dynamic_epsilon/mabrollout')
> data={}
> for v in ['v1','v2']:
>  e=EventAccumulator(str(base/v),size_guidance={'scalars':0}).Reload(); data[v]=e
>  print(v)
>  for tag in e.Tags()['scalars']:
>   a=e.Scalars(tag); print(tag,len(a),'first',[(x.step,round(x.value,6)) for x in a[:3]],'last',[(x.step,round(x.value,6)) for x in a[-2:]])
> for tag in set(data['v1'].Tags()['scalars']) & set(data['v2'].Tags()['scalars']):
>  a=data['v1'].Scalars(tag); b=data['v2'].Scalars(tag)
>  mismatch=next(((x.step,x.value,y.step,y.value) for x,y in zip(a,b) if (x.step,x.value)!=(y.step,y.value)),None)
>  print('FIRST DIFF',tag,mismatch,'overlap',min(len(a),len(b)))
> PY`
> - Ran `nl -ba project/IncompleteBaseline/algorithms/dqn/dqn_agent.py | sed -n '245,267p'
> git -C project/IncompleteBaseline show HEAD:algorithms/dqn/dqn_agent.py | nl -ba | sed -n '220,249p'
> ls -lh project/IncompleteBaseline/dev/logs/qdn/maze5x5/dynamic_epsilon/mabrollout/v*/*`
>
> </details>
</details>

**找到了：这次改动改变了 MAB 在训练阶段的更新行为，所以不是单纯的模块化，也不是 seed 42 失效。**

对比 Git 中的旧版 `722c983` 和当前代码：

| 阶段 | 旧版 | 当前版本 |
|---|---|---|
| warmup | MAB 选动作并更新 Q、N | 相同 |
| ε-exploration | **只选动作，不更新 Q、N** | **选动作后继续更新 Q、N** |

关键在当前 [dqn_agent.py 第 254 行](./project/IncompleteBaseline/algorithms/dqn/dqn_agent.py:254)：

```python
self.eps_exp_strategy.step(state=state)
```

旧版这里调用的是 `action_selection()`，随后由 agent 执行 `env.step()`。新版 `step()` 还会根据反馈更新 MAB，因此后续动作会改变，replay buffer 和训练结果也随之改变。

**我做了短对照，确认这个差异确实能改变动作序列。**相同 seed、相同初始网络、相同环境，warmup 的 8 个动作完全一致；随后强制进入探索：

```text
旧版：0, 0, 0, 0, 0    N 保持不变
新版：0, 1, 2, 3, 0    N 持续更新
```

你的原始 TensorBoard 数据也支持“warmup 后才开始分歧”：

- warmup 期间的 `debug/num_unique_states` 序列完全一致。
- 该指标第一次不同：环境步 **819400**，即训练阶段第 **201** 步。
- `training/loss` 前 **234** 步完全一致，第 **235** 步首次不同。

这不是截图渲染造成的差异，原始日志确实不同。不过，读取时 v2 只训练到约 **7.4 万**个训练步，v1 已到约 **41 万**步，不能直接把两条曲线的末尾成绩相比。

**如果目标是复现旧版，需要恢复“warmup 更新 MAB，训练中的探索只读取 MAB”的行为。**如果你希望探索时继续学习 MAB，那么当前实现符合这个目标，但它应作为一个新的算法变体来比较。

我之前确认了分支能正常运行，却没有对照旧版检查行为是否等价，这一点漏掉了。本次没有修改文件，也没有重跑完整训练。

> 帮我回到那个版本

<details><summary>2 previous messages</summary>

> 我会把 `main.py`、`dqn_agent.py` 和 `rollout.py` 恢复到刚才对比的 `722c983` 版本，恢复“warmup 更新 MAB，训练中的探索只读取 MAB”的行为。训练日志和运行脚本会保留。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `git -C project/IncompleteBaseline diff --stat
>  git -C project/IncompleteBaseline rev-parse HEAD
> rg --files --hidden -g AGENTS.md /Users/jay/Desktop/RL`
> - Ran `python3 - <<'PY'
> from pathlib import Path
> import subprocess,tempfile,ast
> repo=Path('/Users/jay/Desktop/RL/project/IncompleteBaseline')
> paths=['algorithms/common/rollout.py','algorithms/dqn/dqn_agent.py','dev/main.py']
> backup=Path(tempfile.mkdtemp(prefix='mab-before-restore-'))
> for name in paths:
>     p=repo/name
>     b=backup/name;b.parent.mkdir(parents=True,exist_ok=True);b.write_bytes(p.read_bytes())
>     old=subprocess.check_output(['git','-C',str(repo),'show','722c983:'+name])
>     ast.parse(old)
>     p.write_bytes(old)
>     assert p.read_bytes()==old
> print('Backup:',backup)
> print('Restored and syntax checked:',*paths,sep='\n')
> PY
>  git -C project/IncompleteBaseline diff --stat`
>
> </details>
</details>

已将 `main.py`、`dqn_agent.py` 和 `rollout.py` 恢复到 `722c983`，逐文件确认一致，语法检查通过。日志和运行脚本保留。

旧版固定使用 MAB，运行时需要删掉这两个参数，否则会报参数不识别：

```text
--epsilon-exploration-strategy "MABRollout"
--random-rollout-stategy "MABRollout"
```

再次训练请使用新的日志目录，避免混入旧记录。

> 我现在的改动是不是和原版的逻辑是一样的？
>
> 就是在 epsilon-greedy 的时候，如果选择探索的话，就只选择动作，不更新 MAB？

<details><summary>2 previous messages</summary>

> 我会检查你最新的探索分支和 MAB 方法，确认选动作后是否还会更新 Q、N。
>
> <details><summary>Ran a command</summary>
>
> - Ran `rg -n 'MAB|eps_exp|action_selection|def rollout|def random_rollout|preferences|n\[action\]|q\[action\]' project/IncompleteBaseline/algorithms/dqn/dqn_agent.py project/IncompleteBaseline/algorithms/common/rollout.py
> cat project/IncompleteBaseline/algorithms/common/rollout.py
> sed -n '235,305p' project/IncompleteBaseline/algorithms/dqn/dqn_agent.py
>  git -C project/IncompleteBaseline diff -- algorithms/dqn/dqn_agent.py algorithms/common/rollout.py`
>
> </details>
</details>

**是的，当前 ε-greedy 的 MAB 探索分支只选择动作，不根据反馈更新 Q、N，和原版这一点一致。**

[dqn_agent.py 第 254 行](./project/IncompleteBaseline/algorithms/dqn/dqn_agent.py:254) 调用的是：

```python
action = self.eps_exp_strategy.action_selection(state=state)
```

随后由 agent 执行环境动作、保存 transition，没有调用 MAB 的更新逻辑。

不过，当前还有两点需要注意：

1. **warmup 会报错。**你把 `key` 的定义移进了 `action_selection()`，但 [rollout.py 第 54 行](./project/IncompleteBaseline/algorithms/common/rollout.py:54) 的 `step()` 仍然使用 `key`。函数之间不共享局部变量，需要在 `step()` 开头补回：
   ```python
   key = tuple(state.detach().cpu().reshape(-1).tolist())
   ```

2. **遇到新状态时，行为比原版多了一步初始化。**现在 `action_selection()` 会创建零值 Q、N；原版会直接查表，新状态不存在就报 `KeyError`。因此准确地说是：**已有状态的 Q、N 不更新，新状态会初始化。**

在两个策略都设为 `"MABRollout"`、seed 为 42、探索状态已在表中的前提下，修复上述 `key` 问题后，这部分行为与原版一致。

> 还是不对
>
> User context
>
> Images:
> - `/var/folders/kw/d1s4p1k565q1hwqndgb055bh0000gn/T/codex-clipboard-036706e8-2359-4bc1-8d90-de35f6312ef2.png`

<details><summary>8 previous messages</summary>

> 我会重新检查最新代码和 v1、v2 目录里的每个事件文件，确认这次运行从哪一步开始不同，也排查是否混入了之前的日志。
>
> <details><summary>Ran 6 commands</summary>
>
> - Ran `git -C project/IncompleteBaseline diff -- algorithms/common/rollout.py algorithms/dqn/dqn_agent.py dev/main.py
> ls -lh project/IncompleteBaseline/dev/logs/qdn/maze5x5/dynamic_epsilon/mabrollout/v*/*
> rg -n 'EventAccumulator|TensorBoard merges' /Users/jay/.codex/memories/MEMORY.md | head -5`
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B -u - <<'PY'
> from tensorboard.compat.proto.event_pb2 import Event
> from pathlib import Path
> import struct
> base=Path('project/IncompleteBaseline/dev/logs/qdn/maze5x5/dynamic_epsilon/mabrollout')
> data={}
> for v in ['v1','v2']:
>  d={}; p=next((base/v).glob('events*'))
>  with p.open('rb') as f:
>   while True:
>    h=f.read(12)
>    if len(h)<12:break
>    n=struct.unpack('<Q',h[:8])[0];raw=f.read(n);crc=f.read(4)
>    if len(raw)!=n or len(crc)!=4:break
>    e=Event.FromString(raw)
>    for val in e.summary.value:
>     if val.HasField('simple_value'):d.setdefault(val.tag,[]).append((e.step,val.simple_value))
>  data[v]=d
>  print(v,p.name,flush=True)
>  for tag in ['debug/num_unique_states','training/loss','training/success_rate','eval/success_rate']:
>   a=d.get(tag,[]);print(tag,len(a),'first',a[:3],'last',a[-2:],flush=True)
> for tag in data['v1'].keys() & data['v2'].keys():
>  a=data['v1'][tag];b=data['v2'][tag];diff=next(((x,y) for x,y in zip(a,b) if x!=y),None);print('DIFF',tag,diff,'overlap',min(len(a),len(b)),flush=True)
> PY`
> - Ran `stat -f '%Sm %N' -t '%Y-%m-%d %H:%M:%S' project/IncompleteBaseline/algorithms/dqn/dqn_agent.py project/IncompleteBaseline/algorithms/common/rollout.py
> cat project/IncompleteBaseline/dev/maze5x5/run_dqn_dyn.sh`
> - Ran `ps -axo pid,lstart,command`
> - Ran `git -C project/IncompleteBaseline show HEAD:algorithms/common/rollout.py
>  git -C project/IncompleteBaseline log -1 --format=fuller
> cat project/IncompleteBaseline/algorithms/common/replay_buffer.py | head -110
> sed -n '240,302p' project/IncompleteBaseline/envs/procedual_maze/env.py`
> - Ran `git -C project/IncompleteBaseline show 722c983^:algorithms/dqn/dqn_agent.py | sed -n '205,250p'
> git -C project/IncompleteBaseline reflog -5 --date=iso
> /Users/jay/miniconda3/envs/rl/bin/python - <<'PY'
> from datetime import datetime
> for t in [1788806462,1788812466]:print(datetime.fromtimestamp(t).astimezone().isoformat())
> PY`
>
> </details>
>
> 这次差异更早：训练开始后的第 3 步，状态覆盖记录就已不同；两个策略参数也确实传入了当前进程。我会做一个对照：保持相同的 warmup 数据，分别测试“训练探索只读取 MAB”和“训练探索用 uniform”，看哪一种匹配 v1 的前几百步。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `git -C project/IncompleteBaseline show eec5132:algorithms/dqn/dqn_agent.py | sed -n '220,235p'
> git -C project/IncompleteBaseline show cbeda73:algorithms/dqn/dqn_agent.py | sed -n '220,235p'`
> - Ran `cat > /tmp/check_mab_repro.py <<'PY'
> import sys,copy,random,logging,torch,numpy as np,collections
> from unittest.mock import patch
> sys.path.insert(0,'/Users/jay/Desktop/RL/project')
> from IncompleteBaseline.dev.main import create_environments
> from IncompleteBaseline.algorithms.dqn.dqn_agent import DQNAgent
> from IncompleteBaseline.algorithms.common.exploration_rate_calculation import StepDecay
> from tensorboard.compat.proto.event_pb2 import Event
> import struct
> from pathlib import Path
> ref={}
> p=next(Path('/Users/jay/Desktop/RL/project/IncompleteBaseline/dev/logs/qdn/maze5x5/dynamic_epsilon/mabrollout/v1').glob('events*'))
> with p.open('rb') as f:
>  while h:=f.read(12):
>   n=struct.unpack('<Q',h[:8])[0];e=Event.FromString(f.read(n));f.read(4)
>   for v in e.summary.value:
>    if v.tag=='training/loss' and e.step<=300:ref[e.step]=v.simple_value
> random.seed(42);np.random.seed(42);torch.manual_seed(42)
> tr,ev=create_environments(5,5,42,logging.getLogger(),2500,25)
> with patch('IncompleteBaseline.algorithms.dqn.dqn_agent.SummaryWriter'):
>  a=DQNAgent(45,4,42,500000,tr,ev,64,epsilon_strategy=StepDecay(1.,.05,204800),gamma=.99,total_time_steps=1228800,learning_start=819200,training_freq=1,grad_step_per_train=1,random_rollout_stategy='MABRollout',eps_exp_strategy='MABRollout')
> s,_=tr.reset(options={'is_evaluation':False});tr.unwrapped.train_step_limitation=5000
> for t in range(1,819200):
>  s,r,term,trunc,_=a.random_rollout(s)
>  if term or trunc:s,_=tr.reset(options={'is_evaluation':False})
>  if t%100000==0:print('warmup',t,flush=True)
> print('warmup complete',flush=True)
> tr.unwrapped.train_step_limitation=2500
> for mode in ['mab','uniform']:
>  b=copy.copy(a);b.training_env=copy.deepcopy(tr);b._rng=copy.deepcopy(a._rng)
>  b.replaybuffer=copy.copy(a.replaybuffer);b.replaybuffer.pool=collections.deque(a.replaybuffer.pool,maxlen=500000);b.replaybuffer._rng=copy.deepcopy(a.replaybuffer._rng)
>  b.qnetwork=copy.deepcopy(a.qnetwork);b.q_target_network=copy.deepcopy(a.q_target_network);b.optim=torch.optim.Adam(b.qnetwork.parameters())
>  b.eps_exp_strategy=copy.copy(a.eps_exp_strategy) if mode=='mab' else None
>  if b.eps_exp_strategy:
>   b.eps_exp_strategy.preferences=dict(a.eps_exp_strategy.preferences);b.eps_exp_strategy._rng=copy.deepcopy(a.eps_exp_strategy._rng)
>  state=s.clone();first=None;solves=[]
>  for t in range(1,301):
>   state,r,term,trunc,_=b.rollout(t,state);loss=b.update_qnetwork();b.update_target_network()
>   rounded=np.float32(loss).item()
>   if first is None and rounded!=ref[t]:first=(t,rounded,ref[t])
>   if term:solves.append(t)
>   if term or trunc:state,_=b.training_env.reset(options={'is_evaluation':False})
>  print(mode,'first loss mismatch',first,'solved',solves,flush=True)
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B -u /tmp/check_mab_repro.py`
>
> </details>
>
> 短对照复现了当前 v2 的第 235 步 loss，但没有复现 v1；把探索改成 uniform 也不匹配。因此，我之前把 `722c983` 当作 v1 的确切运行版本，证据还不够。我会继续核对探索率：日志目录叫 `dynamic_epsilon`，并不能证明当时实际启用了 `StepDecay`。
>
> <details><summary>Explored 2 files, 2 searches, ran 2 commands</summary>
>
> <details><summary>Explored 2 searches</summary>
>
> - Searched for `mabrollout/v1|epsilon-strategy|epsilon-decay` in `.zsh_history`
> - Searched for `IncompleteBaseline/(algorithms/dqn/dqn_agent.py|algorithms/common/rollout.py|dev/main.py)` in `History`
>
> </details>
>
> - Ran `python3 - <<'PY'
> import json
> from pathlib import Path
> from datetime import datetime
> root=Path('/Users/jay/Library/Application Support/Code/User/History')
> for k in ['-205ec3af','-7e5c2df9','35ed5981','33dedb42']:
>  d=json.loads((root/k/'entries.json').read_text());print(k,d['resource'])
>  for e in d['entries']:
>   if 1788790000000<e['timestamp']<1788814000000:print(e['id'],datetime.fromtimestamp(e['timestamp']/1000).isoformat())
> PY`
>
> <details><summary>Explored 2 files</summary>
>
> - Read `~/Library/Application Support/Code/User/History/35ed5981/5xhO.py`
> - Read `~/Library/Application Support/Code/User/History/-7e5c2df9/kocL.py`
>
> </details>
>
> - Ran `python3 - <<'PY'
> p='/tmp/check_mab_repro.py'
> s=open(p).read().replace("for mode in ['mab','uniform']:","for mode in ['mab_fixed', 'uniform_fixed', 'mab_greedy', 'mab_c0', 'mab_c1', 'mab_update']:")
> s=s.replace("if mode=='mab' else None","if mode!='uniform_fixed' else None")
> s=s.replace("state=s.clone();first=None;solves=[]", """if mode.endswith('fixed'):
>   from IncompleteBaseline.algorithms.common.exploration_rate_calculation import ClassicalExploration
>   b.epsilon_strategy=ClassicalExploration()
>  if mode=='mab_greedy':b.epsilon_strategy=StepDecay(0.,0.,1)
>  if mode=='mab_c0':b.eps_exp_strategy.c=0
>  if mode=='mab_c1':b.eps_exp_strategy.c=1
>  if mode=='mab_update':
>   b.eps_exp_strategy.env=b.training_env
>   b.eps_exp_strategy.preferences={k:[q.clone(),n.clone()] for k,(q,n) in a.eps_exp_strategy.preferences.items()}
>   def rollout(t,state):
>    if b._rng.random()<b.epsilon_strategy.step(t):action,ns,r,term,trunc,info=b.eps_exp_strategy.step(state)
>    else:
>     action=b.get_policy(state);ns,r,term,trunc,info=b.training_env.step(action)
>    from IncompleteBaseline.algorithms.common.replay_buffer import Transition
>    b.replaybuffer.push(Transition(state,action,float(r),ns,term,trunc))
>    return ns,r,term,trunc,info
>   b.rollout=rollout
>  state=s.clone();first=None;solves=[]""")
> open('/tmp/check_mab_variants.py','w').write(s)
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B -u /tmp/check_mab_variants.py`
>
> </details>
>
> 找到更关键的历史记录了：VS Code 在 v1 启动前保存的代码，与 `722c983` 不同。那份代码只要存在 `rollout_warmup`，训练阶段就每一步调用 MAB 并更新，完全跳过 ε-greedy 的概率判断。我正在用这份历史逻辑核对 v1 的原始 loss。
>
> <details><summary>Explored a file, a search, ran 2 commands</summary>
>
> <details><summary>Explored a file, a search</summary>
>
> - Read `~/Library/Application Support/Code/User/History/-7e5c2df9/kocL.py`
> - Searched for `rollout_warmup|training_step \+=` in `kocL.py`
>
> </details>
>
> - Ran `python3 - <<'PY'
> from pathlib import Path
> import subprocess,difflib
> old=Path('/Users/jay/Library/Application Support/Code/User/History/-7e5c2df9/kocL.py').read_text()
> head=subprocess.check_output(['git','-C','project/IncompleteBaseline','show','HEAD:algorithms/dqn/dqn_agent.py'],text=True)
> print(''.join(difflib.unified_diff(head.splitlines(True),old.splitlines(True),fromfile='722c983',tofile='20:39 snapshot')))
> PY`
>
> - Ran `python3 - <<'PY'
> p='/tmp/check_mab_variants.py';s=open(p).read();s=s.replace("['mab_fixed', 'uniform_fixed', 'mab_greedy', 'mab_c0', 'mab_c1', 'mab_update']","['mab_update']");s=s.replace("if b._rng.random()<b.epsilon_strategy.step(t):", "if True:");open('/tmp/check_mab_always.py','w').write(s)
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B -u /tmp/check_mab_always.py`
>
> </details>
</details>

**这次找到并验证了：v1 实际运行的代码不是 `722c983`。我之前认错了基准版本，导致回退方向不对，抱歉。**

在 VS Code 的 [20:39 历史版本](<~/Library/Application Support/Code/User/History/-7e5c2df9/kocL.py:229>) 中，训练分支是：

```python
exploration_rate = self.epsilon_strategy.step(current_time_step)
if self.rollout_warmup:
    action, next_state, reward, terminated, truncated, info = (
        self.rollout_warmup.step(state=state)
    )
else:
    # 这里才执行 epsilon-greedy
```

因为 `self.rollout_warmup` 是 MAB 对象，**v1 在训练期间每一步都使用 MAB，并更新 MAB；ε-greedy 根本没有生效，Q-network 不参与训练动作的选择。**评估时仍然使用 Q-network。

我重新生成了 **819199 步 warmup**，固定这份数据做对照：

| 训练采样方式 | 与 v1 原始 loss 对比 |
|---|---|
| 当前：ε-greedy，探索只读取 MAB | 第 235 步开始不同 |
| ε-greedy，探索时更新 MAB | 第 235 步开始不同 |
| 历史逻辑：每一步都用 MAB 并更新 | **前 300 步逐点完全一致** |

历史逻辑的前几次成功时间也与 v1 日志对应上了。这次不是根据曲线猜测。

所以，**“探索只选动作、不更新 MAB”符合你现在描述的设计，但无法复现 v1，因为 v1 用的是“全程 MAB 采样并更新”。**要复现 v1，需要恢复这个行为，而不是继续调整随机种子。

没有修改你的文件或停止当前训练；验证范围是完整 warmup 加前 300 个训练步。

> 验证20000步

<details><summary>4 previous messages</summary>

> 我会沿用刚才验证的历史逻辑，完成相同的 warmup，再训练 20000 步，并按原来的频率执行评估，逐点对比 v1 的 loss 和评估指标。
>
> <details><summary>Explored a file</summary>
>
> - Read `/tmp/check_mab_always.py`
>
> </details>
>
> 819199 步 warmup 已完成，状态覆盖记录与 v1 全部一致。训练前 2000 步的 loss 也逐点一致，目前没有发现差异。
>
> <details><summary>Ran a command</summary>
>
> - Ran `cat > /tmp/verify_mab_20000.py <<'PY'
> import sys,types,random,logging,struct,json,time,hashlib
> from pathlib import Path
> import numpy as np
> import torch
> from tensorboard.compat.proto.event_pb2 import Event
> from tqdm import tqdm
> sys.path.insert(0,'/Users/jay/Desktop/RL/project')
> from IncompleteBaseline.dev.main import create_environments
> from IncompleteBaseline.algorithms.common.exploration_rate_calculation import StepDecay
> base=Path('/Users/jay/Desktop/RL/project/IncompleteBaseline')
> history=Path('/Users/jay/Library/Application Support/Code/User/History')
> def load(path,name,package):
>  m=types.ModuleType(name);m.__package__=package
>  exec(compile(path.read_text(),str(path),'exec'),m.__dict__)
>  return m
> mab=load(history/'35ed5981/5xhO.py','historical_mab','IncompleteBaseline.algorithms.common')
> agent=load(history/'-7e5c2df9/kocL.py','historical_dqn','IncompleteBaseline.algorithms.dqn')
> agent.MABRollout=mab.MABRollout
> agent.tqdm=lambda x:tqdm(x,disable=True)
> ref={}
> for p in (base/'dev/logs/qdn/maze5x5/dynamic_epsilon/mabrollout/v1').glob('events*'):
>  with p.open('rb') as f:
>   while h:=f.read(12):
>    length=struct.unpack('<Q',h[:8])[0];e=Event.FromString(f.read(length));f.read(4)
>    for v in e.summary.value:
>     limit=839199 if v.tag=='debug/num_unique_states' else 20001 if v.tag=='debug/solved_episode' else 20000
>     if v.HasField('simple_value') and e.step<=limit:ref.setdefault(v.tag,{})[e.step]=v.simple_value
> counts={};mismatches={};values={};start=time.time()
> class Writer:
>  def __init__(self,*args,**kwargs):pass
>  def add_scalar(self,tag,value,step):
>   value=np.float32(value).item();counts[tag]=counts.get(tag,0)+1
>   expected=ref.get(tag,{}).get(step)
>   if value!=expected and tag not in mismatches:mismatches[tag]={'step':step,'actual':value,'expected':expected};print('MISMATCH',tag,mismatches[tag],flush=True)
>   if tag.startswith(('eval/','training/')) and tag!='training/loss':values.setdefault(tag,[]).append([step,value,expected])
>   if tag=='debug/num_unique_states' and step%100000==0:print('env step',step,'elapsed',round(time.time()-start),flush=True)
>   if tag=='training/loss' and step%2000==0:print('training step',step,'loss',value,'mismatches',len(mismatches),flush=True)
>  def flush(self):pass
>  def close(self):pass
> agent.SummaryWriter=Writer
> random.seed(42);np.random.seed(42);torch.manual_seed(42)
> tr,ev=create_environments(5,5,42,logging.getLogger(),2500,25)
> a=agent.DQNAgent(input_dim=45,output_dim=4,seed=42,replay_buffer_size=500000,training_env=tr,eval_env=ev,sample_batch_size=64,epsilon_strategy=StepDecay(1.,.05,204800),gamma=.99,total_time_steps=839199,learning_start=819200,training_freq=1,grad_step_per_train=1,num_eval_episodes=100,eval_freq=10000,max_episode_steps_warmup=5000)
> a.train()
> report={'warmup_steps':819199,'training_steps':20000,'counts':counts,'expected_counts':{k:len(v) for k,v in ref.items()},'first_mismatches':mismatches,'evaluation_values':values,'elapsed_seconds':time.time()-start,'historical_agent':str(history/'-7e5c2df9/kocL.py'),'historical_mab':str(history/'35ed5981/5xhO.py')}
> Path('/tmp/mab_20000_report.json').write_text(json.dumps(report,indent=2))
> print(json.dumps(report,indent=2),flush=True)
> assert not mismatches
> assert counts==report['expected_counts']
> tr.close();ev.close()
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B -u /tmp/verify_mab_20000.py`
>
> </details>
</details>

**已验证 20000 个训练步，与 v1 原始日志逐点完全一致。**

实际运行了 **819199 步 warmup + 20000 步训练**，耗时约 123 秒：

- 20000 个 loss：全部完全一致。
- 839199 个状态覆盖记录：全部一致。
- 153 个成功 episode 记录：全部一致。
- 第 10000、20000 步的训练集及评估集 return、steps、success rate：全部一致。

这确认了：**v1 对应的是“训练期间每一步都使用 MAB 并更新”，而非 ε-greedy 探索时只读取 MAB。**

[验证报告](/tmp/mab_20000_report.json)。没有修改项目代码或原始日志。

<details><summary>39 previous messages</summary>

> 我会以当前代码和日志为准，先定位 ε-greedy 从哪里开始改变采样，再用相同 warmup、相同初始网络做对照实验。结果会同时给出学习曲线、采样覆盖和动作行为的数据，并区分已验证的原因与仍需验证的解释。
>
> <details><summary>Explored 3 files, a list, ran a command</summary>
>
> - Ran `git -C project/IncompleteBaseline status --short
> rg -n 'replay|controlled|TensorBoard merges' /Users/jay/.codex/memories/MEMORY.md | head -12
> rg --files project/IncompleteBaseline/dev/logs/qdn/maze5x5
> cat project/IncompleteBaseline/algorithms/common/rollout.py
> sed -n '95,118p' project/IncompleteBaseline/algorithms/dqn/dqn_agent.py
> sed -n '245,300p' project/IncompleteBaseline/algorithms/dqn/dqn_agent.py`
>
> <details><summary>Explored 3 files, a list</summary>
>
> - Read `./project/IncompleteBaseline/envs/procedual_maze/env.py`
> - Read `./project/IncompleteBaseline/algorithms/common/qnetwork.py`
> - Read `./project/IncompleteBaseline/algorithms/common/exploration_rate_calculation.py`
> - Listed files in `mab_20000_report.json`
>
> </details>
>
> </details>
>
> 当前工作区已经改成“训练全程使用 MAB”，因此不能直接用它代表 ε-greedy。实验会单独固定三种行为：全程 MAB、ε-greedy 探索时更新 MAB、ε-greedy 探索时冻结 MAB。三组共用同一份 warmup 起点，并记录撞墙、往返、重复采样和固定迷宫成功率，先比较 120000 个训练步。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `tail -5 analysis/epsilon_mab_20260908/run.log
> cat analysis/epsilon_mab_20260908/manifest.json`
> - Ran `cat > analysis/epsilon_mab_20260908/summarize.py <<'PY'
> import json,struct
> from pathlib import Path
> import numpy as np
> from tensorboard.compat.proto.event_pb2 import Event
> import matplotlib
> matplotlib.use('Agg')
> import matplotlib.pyplot as plt
> P=Path(__file__).resolve().parent
> plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':160})
> names={'all_mab':'MAB every step + update','eps_live':'epsilon-greedy + live MAB','eps_frozen':'epsilon-greedy + frozen MAB','eps_mask':'epsilon-greedy + live MAB, valid greedy'}
> colors={'all_mab':'#007f73','eps_live':'#d56b25','eps_frozen':'#7754b3','eps_mask':'#2a73b8'}
> modes=[m for m in names if (P/f'{m}_windows.json').exists()]
> fig,axs=plt.subplots(2,2,figsize=(12,7.8),layout='constrained')
> for m in modes:
>  w=json.loads((P/f'{m}_windows.json').read_text());x=np.array([r['step'] for r in w])/1000
>  for ax,ys in zip(axs.flat,[[r['selfloop']/r['steps'] for r in w],[r['backtrack']/max(1,r['backtrack_eligible']) for r in w],[r['unique_states'] for r in w],[r['solved']/r['steps']*1000 for r in w]]):ax.plot(x,ys,label=names[m],color=colors[m],lw=2)
> for ax,title in zip(axs.flat,['No movement (wall / boundary)','Immediate return to previous state','Distinct states per 5,000 new transitions','Goals reached per 1,000 sampling steps']):ax.set(title=title,xlabel='Training steps (thousands)')
> for ax in axs.flat[:2]:ax.set_ylim(0,1)
> fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
> fig.suptitle('What enters the replay buffer?  Seed 42, identical warmup',fontsize=15)
> fig.savefig(P/'sampling.png');fig.savefig(P/'sampling.pdf');plt.close(fig)
> fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
> for m in modes:
>  e=json.loads((P/f'{m}_eval.json').read_text());x=np.array([r['step'] for r in e])/1000
>  for ax,split in zip(axs[:2],['train','heldout']):ax.plot(x,[r['fixed'][split]['success'] for r in e],label=names[m],color=colors[m],lw=2)
>  w=json.loads((P/f'{m}_windows.json').read_text());axs[2].plot(np.array([r['step'] for r in w])/1000,[r['loss_mean'] for r in w],color=colors[m],label=names[m],lw=2)
> for ax,title in zip(axs,['Fixed 100 training mazes','Fixed 100 held-out mazes','Mean TD loss (5,000-step windows)']):ax.set(title=title,xlabel='Training steps (thousands)')
> for ax in axs[:2]:ax.set_ylim(0,1);ax.set_ylabel('Greedy success rate (25-step limit)')
> axs[2].set_yscale('log');fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
> fig.suptitle('Same starting network, replay, environment and RNG state',fontsize=15)
> fig.savefig(P/'learning.png');fig.savefig(P/'learning.pdf');plt.close(fig)
> fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
> for m in modes:
>  w=json.loads((P/f'{m}_windows.json').read_text());x=np.array([r['step'] for r in w])/1000
>  for ax,b in zip(axs,['explore','greedy']):ax.plot(x,[r.get(b+'_selfloop',0)/r[b+'_steps'] if r.get(b+'_steps',0) else np.nan for r in w],label=names[m],color=colors[m],lw=2)
> for ax,b in zip(axs,['MAB exploration actions','Q-network greedy actions']):ax.set(title=b,xlabel='Training steps (thousands)',ylabel='No-movement fraction',ylim=(0,1))
> fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
> fig.savefig(P/'branches.png');plt.close(fig)
> summary={}
> for m in modes:
>  w=json.loads((P/f'{m}_windows.json').read_text());e=json.loads((P/f'{m}_eval.json').read_text());last=w[-4:]
>  summary[m]={'steps':w[-1]['step'],'fixed_final':e[-1],'last20k':{'selfloop':sum(r['selfloop'] for r in last)/sum(r['steps'] for r in last),'backtrack':sum(r['backtrack'] for r in last)/sum(r['backtrack_eligible'] for r in last),'unique_states_mean':np.mean([r['unique_states'] for r in last]).item(),'goals':sum(r['solved'] for r in last),'loss_mean':np.mean([r['loss_mean'] for r in last]).item()}}
> (P/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
> PY
> tail -5 analysis/epsilon_mab_20260908/run.log`
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B - <<'PY'
> from pathlib import Path
> from tensorboard.compat.proto.event_pb2 import Event
> import struct,numpy as np
> p=next(Path('project/IncompleteBaseline/dev/logs/qdn/maze5x5/dynamic_epsilon/mabrollout/v1').glob('events*'))
> a=np.load('analysis/epsilon_mab_20260908/all_mab_loss.npy');bad=[];n=0
> with p.open('rb') as f:
>  while h:=f.read(12):
>   z=struct.unpack('<Q',h[:8])[0];e=Event.FromString(f.read(z));f.read(4)
>   for v in e.summary.value:
>    if v.tag=='training/loss' and 1<=e.step<=len(a):
>     n+=1
>     if v.simple_value!=a[e.step-1]:bad.append(e.step)
> print('Baseline comparison:',n,'loss points;',len(bad),'mismatches')
> PY
> tail -4 analysis/epsilon_mab_20260908/run.log`
>
> </details>
>
> 实验的 warmup 已完成，三组起点包含同样的 92718 个已访问状态。除了原来的随机评估，我增加了固定的 100 个训练迷宫和 100 个未见迷宫，避免“每次抽到不同迷宫”影响成功率比较。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `tail -4 analysis/epsilon_mab_20260908/run.log`
> - Ran `head -35 project/IncompleteBaseline/envs/procedual_maze/env.py
> rg -n 'WALL\s*=|_action_to_direction' project/IncompleteBaseline/envs/procedual_maze -g '*.py' | head -12
> ls -lh analysis/epsilon_mab_20260908/*weights*`
> - Ran `python3 - <<'PY'
> from pathlib import Path
> import hashlib,json
> out=Path('analysis/epsilon_mab_20260908');root=Path('project/IncompleteBaseline');manifest=json.loads((out/'manifest.json').read_text())
> for rel,h in manifest['source_hashes'].items():
>  p=root/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
>  dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
> p=out/'experiment.py';s=p.read_text().replace('env.current_maze[tuple(c)]!=0','env.current_maze[tuple(c)]!=1');p.write_text(s)
> PY
> tail -2 analysis/epsilon_mab_20260908/run.log`
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log`
>
> </details>
>
> 全程 MAB 对照组已跑到 35000 步；新增诊断没有改变训练，已核对的前 15000 个 loss 与 v1 完全一致。一个值得区分的现象是：MAB 本身也有约 46%–48% 的原地不动动作，所以不能仅凭“撞墙多”就断定 ε-greedy 的问题；还要比较两组是否更容易陷入重复往返、是否持续获得成功轨迹。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `cat > analysis/epsilon_mab_20260908/policy_audit.py <<'PY'
> import sys,json,logging,collections
> from pathlib import Path
> import numpy as np
> import torch
> ROOT=Path('/Users/jay/Desktop/RL/project');sys.path.insert(0,str(ROOT))
> from IncompleteBaseline.dev.main import create_environments
> from IncompleteBaseline.algorithms.common.qnetwork import QNetwork
> from IncompleteBaseline.envs.procedual_maze.util import WALL,PLAYER,EMPTY,GOAL
> P=Path(__file__).resolve().parent
> if not (P/'probe_states.npz').exists():
>  tr,ev=create_environments(5,5,42,logging.getLogger(),2500,25);env=ev.unwrapped
>  states=[];wall=[];optimal=[];exact=[];splits=[];mazeids=[]
>  for split in [False,True]:
>   for mi in range(100):
>    ev.reset(options={'is_evaluation':split,'maze_index':mi});grid=env.current_maze.copy();grid[grid==PLAYER]=EMPTY;goal=tuple(env._goal_location)
>    dist={goal:0};queue=collections.deque([goal])
>    while queue:
>     p=queue.popleft()
>     for delta in env._action_to_direction.values():
>      v=tuple(np.array(p)+delta)
>      if 0<=v[0]<5 and 0<=v[1]<5 and grid[v]!=WALL and v not in dist:dist[v]=dist[p]+1;queue.append(v)
>    def value(d):return 0. if d==0 else -.01*(1-.99**(d-1))/.01+.99**(d-1)
>    for pos,d in dist.items():
>     if d==0:continue
>     env.current_maze=grid.copy();env.current_maze[pos]=PLAYER;env._agent_location=np.array(pos,dtype=np.int32)
>     obs=ev.observation(env._get_observation()).reshape(-1).numpy();wa=[];op=[];qs=[]
>     for delta in env._action_to_direction.values():
>      nxt=tuple(np.array(pos)+delta);invalid=nxt not in dist
>      wa.append(invalid);op.append(not invalid and dist[nxt]==d-1)
>      qs.append(-.05+.99*value(d) if invalid else 1. if nxt==goal else -.01+.99*value(dist[nxt]))
>     states.append(obs);wall.append(wa);optimal.append(op);exact.append(qs);splits.append(split);mazeids.append(mi)
>  np.savez_compressed(P/'probe_states.npz',states=states,wall=wall,optimal=optimal,exact=exact,splits=splits,mazeids=mazeids)
> d=np.load(P/'probe_states.npz');s=torch.tensor(d['states']);results={}
> for file in P.glob('*_latest_weights.pt'):
>  mode=file.name.replace('_latest_weights.pt','');q=QNetwork(45,4);q.load_state_dict(torch.load(file,weights_only=True)['q']);q.eval()
>  with torch.no_grad():pred=q(s).numpy()
>  act=pred.argmax(1);idx=np.arange(len(act));per={}
>  for split in [False,True]:
>   mask=d['splits']==split;i=idx[mask];ac=act[mask]
>   per['heldout' if split else 'train']={'states':len(i),'greedy_invalid_fraction':float(d['wall'][i,ac].mean()),'greedy_shortest_fraction':float(d['optimal'][i,ac].mean()),'q_mae':float(np.abs(pred[mask]-d['exact'][mask]).mean()),'chosen_q_bias':float((pred[i,ac]-d['exact'][i,ac]).mean())}
>  results[mode]=per
> (P/'policy_audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
> PY
> tail -2 analysis/epsilon_mab_20260908/run.log`
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/policy_audit.py`
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log`
>
> </details>
>
> 全程 MAB 已到 70000 步。固定状态检查显示，此时网络在未见迷宫中仍有约 21% 的状态会贪心选择无效动作；这只是一个可能的薄弱点，还不能解释组间差异。接下来会看 ε-greedy 是否把采样集中到这些状态，以及冻结 MAB 是否另行造成往返循环。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log`
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log`
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log`
>
> </details>
>
> 对照组已超过 100000 步，原来的成功率上升趋势也出现了。接下来两组将从相同的 warmup 检查点重新开始，而不是接着对照组训练；这样差异才能归因于采样策略，而不是初始网络或 replay buffer 不同。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `tail -4 analysis/epsilon_mab_20260908/run.log`
> - Ran `tail -4 analysis/epsilon_mab_20260908/run.log`
> - Ran `cat > analysis/epsilon_mab_20260908/verify_reference.py <<'PY'
> from pathlib import Path
> from tensorboard.compat.proto.event_pb2 import Event
> import struct,numpy as np,json
> p=Path(__file__).resolve().parent
> src=next(Path('/Users/jay/Desktop/RL/project/IncompleteBaseline/dev/logs/qdn/maze5x5/dynamic_epsilon/mabrollout/v1').glob('events*'))
> ref={}
> with src.open('rb') as f:
>  while h:=f.read(12):
>   n=struct.unpack('<Q',h[:8])[0];e=Event.FromString(f.read(n));f.read(4)
>   for v in e.summary.value:
>    if v.HasField('simple_value'):ref.setdefault(v.tag,{})[e.step]=v.simple_value
> loss=np.load(p/'all_mab_loss.npy');mismatch=[i+1 for i,x in enumerate(loss) if x!=ref['training/loss'][i+1]]
> evals=json.loads((p/'all_mab_eval.json').read_text());bad=[];count=0
> for e in evals:
>  if not e['step']:continue
>  for split,label in [('train','training'),('heldout','eval')]:
>   for metric,tag in [('success','success_rate'),('return','return'),('steps','steps')]:
>    actual=np.float32(e['official'][split][metric]).item();expected=ref[label+'/'+tag][e['step']];count+=1
>    if actual!=expected:bad.append([e['step'],label,metric,actual,expected])
> r={'loss_points':len(loss),'loss_mismatches':mismatch,'evaluation_points':count,'evaluation_mismatches':bad,'source':str(src)}
> (p/'reference_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));assert not mismatch and not bad
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/verify_reference.py`
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log`
>
> </details>
>
> 对照组的 120000 个 loss 和 72 个原始评估数值，已全部与 v1 精确匹配。ε-greedy 且持续更新 MAB 的组也已开始；目前前 10000 步的采样覆盖与对照接近，还没有证据说明 ε-greedy 本身一定更差。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log`
> - Ran `python3 - <<'PY'
> import json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> for m in ['all_mab','eps_live']:
>  w=json.loads((p/f'{m}_windows.json').read_text());r=w[-1]
>  print(m,'step',r['step'],'backtrack',r['backtrack']/r['backtrack_eligible'],'selfloop',r['selfloop']/r['steps'],'greedy selfloop',r.get('greedy_selfloop',0)/max(1,r.get('greedy_steps',0)),'explore selfloop',r.get('explore_selfloop',0)/r['explore_steps'])
> PY`
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log`
>
> </details>
>
> 到 45000 步时，持续更新 MAB 的 ε-greedy 组没有表现出更严重的撞墙，最近窗口反而采到了更多不同状态和成功轨迹。因此，“贪心动作导致撞墙增多”暂时不受数据支持；我会重点看冻结 MAB 的组是否才是此前下降的主要来源。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log`
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log`
> - Ran `python3 - <<'PY'
> import json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> for mode in ['all_mab','eps_live']:
>  rows=[r for r in json.loads((p/f'{mode}_windows.json').read_text()) if 50000<r['step']<=70000]
>  print(mode,{k:sum(r[k] for r in rows) for k in ['steps','selfloop','backtrack','solved']},'unique',[r['unique_states'] for r in rows])
> PY`
>
> </details>
>
> 现在出现了可量化的差异。在第 50001–70000 步这个相同区间，全程 MAB 到达目标 167 次，ε-greedy＋更新 MAB 只有 74 次；即时往返从 1072 次增加到 2402 次。两组原地不动总数却很接近，说明问题更像是反复走回头路、困在少数状态，而不只是撞墙。
>
> <details><summary>Ran a command</summary>
>
> - Ran `sed -n '62,103p' analysis/epsilon_mab_20260908/intervention.py
> tail -2 analysis/epsilon_mab_20260908/run.log`
>
> </details>
>
> 为检验“贪心动作让轨迹反复折返”是否真是原因，我增加了一个有针对性的对照：只有当贪心动作会立即走回上一状态时，才让 MAB 接管这一步，其余设置保持不变。这是诊断实验，暂不改你的训练代码；如果覆盖和学习一起改善，因果证据会更强。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log
> tail -2 analysis/epsilon_mab_20260908/intervention.log`
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log
> tail -2 analysis/epsilon_mab_20260908/intervention.log`
> - Ran `python3 - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908/summarize.py');s=p.read_text().replace("'eps_mask':'epsilon-greedy + live MAB, valid greedy'","'eps_avoid_return':'epsilon-greedy + reverse-step fallback'").replace("'eps_mask':'#2a73b8'","'eps_avoid_return':'#2a73b8'")
> s=s.replace("w=json.loads((P/f'{m}_windows.json').read_text());x=np.array([r['step'] for r in w])/1000\n for ax,b", "if m=='eps_avoid_return':continue\n w=json.loads((P/f'{m}_windows.json').read_text());x=np.array([r['step'] for r in w])/1000\n for ax,b")
> p.write_text(s)
> PY
> tail -3 analysis/epsilon_mab_20260908/run.log`
>
> </details>
>
> ε-greedy＋更新 MAB 的 120000 步已完成，后段确实出现了采样收缩：最近 5000 步只有 247 个不同状态、到达目标 21 次；全程 MAB 同一区间是 544 个状态、43 次。冻结 MAB 和“遇到立即折返时交给 MAB”的两组还在运行，它们会帮助区分原因。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/summarize.py`
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log
> tail -2 analysis/epsilon_mab_20260908/intervention.log`
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log
> tail -2 analysis/epsilon_mab_20260908/intervention.log`
>
> </details>
>
> 冻结 MAB 的问题已经很明显：到 40000 步只到达目标 33 次，最近 5000 步只有 61 个不同状态。代码也解释了这个现象：Q 和 N 都冻结后，已经尝试过的状态会反复给出同一个 UCB 选择；未尝试动作也永远不会被标记为已尝试。这与正常持续探索的 MAB 不同。
>
> <details><summary>Ran a command</summary>
>
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log
> tail -2 analysis/epsilon_mab_20260908/intervention.log`
>
> </details>
>
> Viewed an image
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log
> tail -2 analysis/epsilon_mab_20260908/intervention.log`
> - Ran `python3 - <<'PY'
> import json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> for m in ['all_mab','eps_live','eps_frozen','eps_avoid_return']:
>  rs=json.loads((p/f'{m}_windows.json').read_text())[-4:]
>  print(m,'to',rs[-1]['step'],'outcomes/state',sum(r['unique_transitions'] for r in rs)/sum(r['unique_states'] for r in rs),'top10',sum(r['top10_fraction'] for r in rs)/len(rs),'trunc',sum(r['truncated'] for r in rs))
> PY`
>
> </details>
>
> 进一步的统计显示，差异不只是“见过多少状态”。最近 20000 步中，ε-greedy＋更新 MAB 的每个 5000 步窗口，最常见的 10 条 transition 平均占到 44.2%；全程 MAB 只有 2.5%。这说明 replay 新数据明显集中在少数重复转移上。我也会检查干预是否真正改善学习，而不只改善采样数字。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `tail -2 analysis/epsilon_mab_20260908/run.log
> tail -2 analysis/epsilon_mab_20260908/intervention.log
> tail -2 analysis/epsilon_mab_20260908/observe_all.log`
> - Ran `python3 - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908');s=(p/'experiment.py').read_text()
> s=s.replace("default=['all_mab','eps_live','eps_frozen']","default=['eps_avoid_return']").replace("OUT/'manifest.json'","OUT/'intervention_manifest.json'")
> s=s.replace("ns,r,term,trunc,info=a.training_env.step(action)\n  nk=key(ns);loop=sk==nk", """env=a.training_env.unwrapped
>    candidate=env._agent_location+env._action_to_direction[action]
>    reverse=False
>    if prev is not None and prev!=sk:
>     previous=np.asarray(prev).reshape(5,9)
>     previous_pos=np.array([previous[:,5].argmax(),previous[:,6].argmax()])
>     reverse=bool(np.array_equal(candidate,previous_pos))
>    if mode=='eps_avoid_return' and reverse:
>     stat['interventions']+=1
>     action,ns,r,term,trunc,info=mab.step(s)
>    else:ns,r,term,trunc,info=a.training_env.step(action)
>   nk=key(ns);loop=sk==nk""")
> compile(s,'intervention.py','exec');(p/'intervention.py').write_text(s)
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B -u analysis/epsilon_mab_20260908/intervention.py > analysis/epsilon_mab_20260908/intervention.log 2>&1`
>
> </details>
>
> 只在贪心立即折返时改用 MAB，虽然增加了成功采样，却还没有恢复评估成功率。这说明“回头路多”不是充分解释。我正在补查一个实现差别：ε-greedy 中由网络选出的动作，当前不会更新 MAB 的计数和反馈；最后一组会保持相同 ε 调度，但让 MAB 观察所有实际动作。
>
> <details><summary>Explored a file, ran 5 commands</summary>
>
> - Ran `mkdir -p /Users/jay/Desktop/RL/analysis/epsilon_mab_20260908
> cp /tmp/verify_mab_20000.py /Users/jay/Desktop/RL/analysis/epsilon_mab_20260908/previous_validation.py
> cat > /Users/jay/Desktop/RL/analysis/epsilon_mab_20260908/experiment.py <<'PY'
> import os,sys,copy,random,logging,json,time,argparse,collections,hashlib
> from pathlib import Path
> import numpy as np
> import torch
> from unittest.mock import patch
> ROOT=Path('/Users/jay/Desktop/RL/project');sys.path.insert(0,str(ROOT))
> from IncompleteBaseline.dev.main import create_environments
> from IncompleteBaseline.algorithms.dqn.dqn_agent import DQNAgent
> from IncompleteBaseline.algorithms.common.exploration_rate_calculation import StepDecay
> from IncompleteBaseline.algorithms.common.replay_buffer import Transition
> OUT=Path(__file__).resolve().parent
> parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=120000);parser.add_argument('--modes',nargs='+',default=['all_mab','eps_live','eps_frozen']);args=parser.parse_args()
> SEED=42
> random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
> start=time.time()
> tr,ev=create_environments(5,5,SEED,logging.getLogger(),2500,25)
> with patch('IncompleteBaseline.algorithms.dqn.dqn_agent.SummaryWriter'):
>  base=DQNAgent(45,4,SEED,500000,tr,ev,64,epsilon_strategy=StepDecay(1.,.05,204800),gamma=.99,total_time_steps=1228800,learning_start=819200,training_freq=1,grad_step_per_train=1,eval_freq=10000,random_rollout_stategy='MABRollout',eps_exp_strategy='MABRollout')
> manifest={'seed':42,'warmup_steps':819199,'train_steps':args.steps,'modes':args.modes,'replay_capacity':500000,'batch_size':64,'train_episode_limit':2500,'warmup_episode_limit':5000,'eval_limit':25,'torch_threads':torch.get_num_threads(),'torch_version':torch.__version__,'source_hashes':{p:hashlib.sha256((ROOT/'IncompleteBaseline'/p).read_bytes()).hexdigest() for p in ['algorithms/common/rollout.py','algorithms/dqn/dqn_agent.py','dev/main.py']}}
> (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
> state,_=tr.reset(options={'is_evaluation':False});tr.unwrapped.train_step_limitation=5000
> for t in range(1,819200):
>  state,_,term,trunc,_=base.random_rollout(state)
>  if term or trunc:state,_=tr.reset(options={'is_evaluation':False})
>  if t%100000==0:print('warmup',t,'seconds',round(time.time()-start),flush=True)
> tr.unwrapped.train_step_limitation=2500
> print('warmup_done','states',len(base.rollout_warmup.preferences),'seconds',round(time.time()-start),flush=True)
> # Freeze the warmup checkpoint in memory; each arm gets independent mutable state.
> def key(s):return tuple(s.reshape(-1).tolist())
> def fixed_eval(a,env):
>  results={}
>  for split in [False,True]:
>   wins=0;returns=[];lengths=[]
>   for i in range(100):
>    s,_=env.reset(options={'is_evaluation':split,'maze_index':i});r_sum=0
>    for j in range(25):
>     action=a.get_policy(s);s,r,term,trunc,_=env.step(action);r_sum+=r
>     if term or trunc:break
>    wins+=int(term);returns.append(r_sum);lengths.append(j+1)
>   results['heldout' if split else 'train']={'success':wins/100,'return':float(np.mean(returns)),'steps':float(np.mean(lengths))}
>  return results
> for mode in args.modes:
>  a=copy.copy(base);a.training_env=copy.deepcopy(tr);a.eval_env=copy.deepcopy(ev);probe_env=copy.deepcopy(ev)
>  a._rng=copy.deepcopy(base._rng);a.epsilon_strategy=copy.deepcopy(base.epsilon_strategy)
>  a.qnetwork=copy.deepcopy(base.qnetwork);a.q_target_network=copy.deepcopy(base.q_target_network);a.optim=torch.optim.Adam(a.qnetwork.parameters())
>  a.replaybuffer=copy.copy(base.replaybuffer);a.replaybuffer.pool=collections.deque(base.replaybuffer.pool,maxlen=500000);a.replaybuffer._rng=copy.deepcopy(base.replaybuffer._rng)
>  mab=copy.copy(base.rollout_warmup);mab.env=a.training_env;mab._rng=copy.deepcopy(base.rollout_warmup._rng);mab.preferences={k:[q.clone(),n.clone()] for k,(q,n) in base.rollout_warmup.preferences.items()}
>  s=state.clone();prev=None;counter=collections.Counter();stat=collections.Counter();windows=[];evals=[];losses=[];total_solved=0;greedy_diag=[]
>  mode_start=time.time();fixed=fixed_eval(a,probe_env);evals.append({'step':0,'fixed':fixed})
>  print('START',mode,fixed,flush=True)
>  for t in range(1,args.steps+1):
>   sk=key(s);eps=a.epsilon_strategy.step(t)
>   explore=True if mode=='all_mab' else a._rng.random()<eps
>   branch='explore' if explore else 'greedy'
>   if explore:
>    if mode=='eps_frozen':action=mab.action_selection(s);ns,r,term,trunc,info=a.training_env.step(action)
>    else:action,ns,r,term,trunc,info=mab.step(s)
>   else:
>    action=a.get_policy(s)
>    if mode=='eps_mask':
>     env=a.training_env.unwrapped
>     pos=env._agent_location
>     valid=[]
>     for act,d in env._action_to_direction.items():
>      c=pos+d
>      if 0<=c[0]<env.height and 0<=c[1]<env.width and env.current_maze[tuple(c)]!=0:valid.append(act)
>     with torch.no_grad():
>      qs=a.qnetwork(s).reshape(-1); action=max(valid,key=lambda x:qs[x].item()) if valid else action
>    ns,r,term,trunc,info=a.training_env.step(action)
>   nk=key(ns);loop=sk==nk
>   stat['steps']+=1;stat[branch+'_steps']+=1;stat['selfloop']+=int(loop);stat[branch+'_selfloop']+=int(loop)
>   stat['backtrack']+=int(prev is not None and nk==prev and not loop);stat['backtrack_eligible']+=int(prev is not None)
>   stat['solved']+=int(term);stat['truncated']+=int(trunc);stat['reward_sum']+=r
>   counter[(sk,action,float(r),nk,bool(term),bool(trunc))]+=1
>   a.replaybuffer.push(Transition(s,action,float(r),ns,term,trunc));loss=a.update_qnetwork();a.update_target_network();losses.append(loss)
>   total_solved+=int(term)
>   # Match historical evaluation order and RNG, separate from fixed-set probes.
>   if t%10000==0:
>    official={}
>    for split in [True,False]:
>     rr,ll,ss=a.evaluation(a.eval_env,evaluation=split);official['heldout' if split else 'train']={'return':float(rr),'steps':float(ll),'success':float(ss)}
>    fixed=fixed_eval(a,probe_env)
>    rec={'step':t,'epsilon':eps,'official':official,'fixed':fixed,'total_solved':total_solved};evals.append(rec)
>    torch.save({'q':a.qnetwork.state_dict(),'target':a.q_target_network.state_dict()},OUT/f'{mode}_latest_weights.pt')
>    (OUT/f'{mode}_eval.json').write_text(json.dumps(evals,indent=2))
>    print('EVAL',mode,t,rec,'elapsed',round(time.time()-mode_start),flush=True)
>   if t%5000==0:
>    rec=dict(stat);rec.update(step=t,epsilon=eps,unique_transitions=len(counter),unique_states=len({x[0] for x in counter}),top10_fraction=sum(c for _,c in counter.most_common(10))/5000,loss_mean=float(np.mean(losses[-5000:])))
>    windows.append(rec);(OUT/f'{mode}_windows.json').write_text(json.dumps(windows,indent=2));counter.clear();stat.clear()
>    np.save(OUT/f'{mode}_loss.npy',np.asarray(losses,dtype=np.float32))
>    print('WINDOW',mode,t,'selfloop',round(rec['selfloop']/5000,3),'unique',rec['unique_states'],'solved',rec['solved'],flush=True)
>   if term or trunc:s,_=a.training_env.reset(options={'is_evaluation':False});prev=None
>   else:prev=sk;s=ns
>  (OUT/f'{mode}_done.json').write_text(json.dumps({'steps':args.steps,'elapsed':time.time()-mode_start,'total_solved':total_solved}))
>  print('DONE',mode,flush=True)
> print('ALL DONE',time.time()-start,flush=True)
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B -u /Users/jay/Desktop/RL/analysis/epsilon_mab_20260908/experiment.py > /Users/jay/Desktop/RL/analysis/epsilon_mab_20260908/run.log 2>&1`
>
> - Ran `cat > analysis/epsilon_mab_20260908/trajectory_audit.py <<'PY'
> import sys,logging,json,collections
> from pathlib import Path
> import torch
> P=Path(__file__).resolve().parent;sys.path.insert(0,'/Users/jay/Desktop/RL/project')
> from IncompleteBaseline.dev.main import create_environments
> from IncompleteBaseline.algorithms.common.qnetwork import QNetwork
> tr,ev=create_environments(5,5,42,logging.getLogger(),2500,25)
> result={}
> for f in P.glob('*_done.json'):
>  mode=f.name.replace('_done.json','');q=QNetwork(45,4);q.load_state_dict(torch.load(P/f'{mode}_latest_weights.pt',weights_only=True)['q']);q.eval();counts={}
>  for split in [False,True]:
>   c=collections.Counter();per=[]
>   for idx in range(100):
>    s,_=ev.reset(options={'is_evaluation':split,'maze_index':idx});seen={tuple(s.reshape(-1).tolist()):0};label='budget';period=None
>    for t in range(1,26):
>     with torch.no_grad():a=q(s).argmax().item()
>     s,r,term,trunc,_=ev.step(a);key=tuple(s.reshape(-1).tolist())
>     if term:label='success';break
>     if key in seen:
>      period=t-seen[key];label='wall_cycle' if period==1 else 'two_state_cycle' if period==2 else 'longer_cycle';break
>     seen[key]=t
>    c[label]+=1;per.append({'maze_index':idx,'result':label,'cycle_period':period})
>   counts['heldout' if split else 'train']={'counts':dict(c),'per_maze':per}
>  result[mode]=counts
> (P/'trajectory_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({m:{k:v['counts'] for k,v in r.items()} for m,r in result.items()},indent=2))
> PY
> python3 - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908/summarize.py');s=p.read_text().replace('import json,struct','import json,struct,os\nos.environ.setdefault("MPLCONFIGDIR","/tmp/mab-mpl-cache")');p.write_text(s)
> PY
> tail -3 analysis/epsilon_mab_20260908/run.log
> tail -3 analysis/epsilon_mab_20260908/intervention.log
> tail -2 analysis/epsilon_mab_20260908/observe_all.log`
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/trajectory_audit.py
> /Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/policy_audit.py`
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/summarize.py > analysis/epsilon_mab_20260908/summary_stdout.json`
>
> <details><summary>Explored a file</summary>
>
> - Read `./analysis/epsilon_mab_20260908/summary.json`
>
> </details>
>
> - Ran `tail -2 analysis/epsilon_mab_20260908/observe_all.log
> python3 - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908/summarize.py');s=p.read_text().replace("'eps_avoid_return':'epsilon-greedy + reverse-step fallback'","'eps_avoid_return':'epsilon-greedy + reverse-step fallback','eps_observe_all':'epsilon-greedy + observe every action'").replace("'eps_avoid_return':'#2a73b8'","'eps_avoid_return':'#2a73b8','eps_observe_all':'#ba4262'")
> s=s.replace("[r['solved']/r['steps']*1000 for r in w]","[r['top10_fraction'] for r in w]").replace("'Goals reached per 1,000 sampling steps'","'Top 10 transitions: fraction of new samples'")
> s=s.replace("if m=='eps_avoid_return':continue","if m=='eps_avoid_return':continue")
> s=s.replace("'loss_mean':np.mean([r['loss_mean'] for r in last]).item()","'loss_mean':np.mean([r['loss_mean'] for r in last]).item(),'top10_fraction':np.mean([r['top10_fraction'] for r in last]).item(),'transitions_per_state':sum(r['unique_transitions'] for r in last)/sum(r['unique_states'] for r in last),'truncations':sum(r['truncated'] for r in last)")
> p.write_text(s)
> PY`
>
> </details>
>
> 四组已完成。对 100 个固定未见迷宫逐条追踪后，普通 ε-greedy 组有 92 个迷宫最终困在“反复选择无效动作”的同一状态，成功 1 个；全程 MAB 训练出的网络成功 26 个。折返干预虽减少了回头路，成功也只有 3 个，所以不能把失败简单归结为两步循环。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `python3 - <<'PY'
> import json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> for m in ['all_mab','eps_live','eps_frozen','eps_avoid_return']:
>  rs=json.loads((p/f'{m}_windows.json').read_text())[-4:]
>  print(m)
>  for b in ['explore','greedy']:
>   n=sum(r.get(b+'_steps',0) for r in rs);stay=sum(r.get(b+'_selfloop',0) for r in rs)
>   print(b,n,stay,stay/n if n else None)
>  print('return interventions',sum(r.get('interventions',0) for r in rs))
> PY`
> - Ran `tail -2 analysis/epsilon_mab_20260908/observe_all.log`
> - Ran `python3 - <<'PY'
> from pathlib import Path
> import hashlib,json
> p=Path('analysis/epsilon_mab_20260908').resolve();src=Path('project/IncompleteBaseline')
> manifest=json.loads((p/'manifest.json').read_text())
> for rel,h in manifest['source_hashes'].items():assert hashlib.sha256((src/rel).read_bytes()).hexdigest()==h
> hashes={}
> for f in src.rglob('*.py'):
>  if any(x in f.parts for x in ['__pycache__','logs']):continue
>  rel=f.relative_to(src);dest=p/'frozen_project/IncompleteBaseline'/rel;dest.parent.mkdir(parents=True,exist_ok=True);data=f.read_bytes();dest.write_bytes(data);hashes[str(rel)]=hashlib.sha256(data).hexdigest()
> (p/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
> for name in ['experiment.py','intervention.py','observe_all.py','policy_audit.py','trajectory_audit.py']:
>  f=p/name;s=f.read_text().replace("ROOT=Path('/Users/jay/Desktop/RL/project')", "ROOT=Path(__file__).resolve().parent/'frozen_project'").replace("sys.path.insert(0,'/Users/jay/Desktop/RL/project')", "sys.path.insert(0,str(P/'frozen_project'))");f.write_text(s)
> print('Frozen source files:',len(hashes))
> PY
> tail -2 analysis/epsilon_mab_20260908/observe_all.log`
>
> </details>
>
> 让 MAB 观察所有动作后，采样覆盖目前保持得更好，但我还在等它到同样的 120000 步，避免提前用局部改善下结论。代码、数据和绘图脚本已经保存成独立实验目录，包含源码快照，后续可以复查和重跑。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `tail -2 analysis/epsilon_mab_20260908/observe_all.log`
> - Ran `tail -2 analysis/epsilon_mab_20260908/observe_all.log`
> - Ran `python3 - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908');s=(p/'experiment.py').read_text()
> s=s.replace("default=['all_mab','eps_live','eps_frozen']","default=['eps_observe_all']").replace("OUT/'manifest.json'","OUT/'observe_all_manifest.json'")
> s=s.replace("nk=key(ns);loop=sk==nk", """nk=key(ns);loop=sk==nk
>   if mode=='eps_observe_all' and not explore:
>    if sk not in mab.preferences:
>     mab.preferences[sk]=[torch.zeros(4,dtype=torch.float32),torch.zeros(4,dtype=torch.int32)]
>    intrinsic=(-1 if nk in mab.preferences else 1)+r
>    q,n=mab.preferences[sk];n[action]+=1;q[action]+=mab.lr*(intrinsic-q[action])""")
> compile(s,'observe_all.py','exec');(p/'observe_all.py').write_text(s)
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B -u analysis/epsilon_mab_20260908/observe_all.py > analysis/epsilon_mab_20260908/observe_all.log 2>&1`
> - Ran `tail -4 analysis/epsilon_mab_20260908/observe_all.log
> /Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/summarize.py > analysis/epsilon_mab_20260908/summary_stdout.json
> /Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/policy_audit.py > analysis/epsilon_mab_20260908/policy_stdout.json
> /Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/trajectory_audit.py > analysis/epsilon_mab_20260908/trajectory_stdout.json`
>
> </details>
>
> 五组都已完成。一个有直接对照支持的改进是：保持相同 ε 调度，只让 MAB 也更新网络所选动作的计数和反馈，固定未见迷宫成功率从 1% 提高到 11%；全程 MAB 是 26%。因此，漏记贪心动作的反馈确实是一个影响因素，但还不能解释全部差距。我正在整理最终图表和可复查的数据。
>
> <details><summary>Explored 2 files, ran a command</summary>
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./analysis/epsilon_mab_20260908/summary.json`
> - Read `./analysis/epsilon_mab_20260908/policy_audit.json`
>
> </details>
>
> - Ran ``cat > analysis/epsilon_mab_20260908/report.py <<'PY'
> from pathlib import Path
> import json,os
> import numpy as np
> os.environ.setdefault('MPLCONFIGDIR','/tmp/mab-mpl-cache')
> import matplotlib
> matplotlib.use('Agg')
> import matplotlib.pyplot as plt
> P=Path(__file__).resolve().parent
> s=json.loads((P/'summary.json').read_text());audit=json.loads((P/'policy_audit.json').read_text());traj=json.loads((P/'trajectory_audit.json').read_text())
> modes=['all_mab','eps_live','eps_frozen','eps_avoid_return','eps_observe_all']
> names=['全程 MAB，持续更新','ε-greedy，仅探索动作更新 MAB','ε-greedy，冻结 MAB','ε-greedy，立即折返时 MAB 接管','ε-greedy，所有动作更新 MAB']
> english=['MAB every step','epsilon + live MAB','epsilon + frozen MAB','epsilon + reverse fallback','epsilon + observe all']
> for m in modes:
>  assert s[m]['steps']==120000
>  assert len(np.load(P/f'{m}_loss.npy'))==120000
>  assert np.isfinite(np.load(P/f'{m}_loss.npy')).all()
>  assert json.loads((P/f'{m}_done.json').read_text())['steps']==120000
>  assert len(json.loads((P/f'{m}_eval.json').read_text()))==13
>  assert sum(traj[m]['heldout']['counts'].values())==100
>  assert traj[m]['heldout']['counts'].get('success',0)==round(s[m]['fixed_final']['fixed']['heldout']['success']*100)
> ref=json.loads((P/'reference_check.json').read_text());assert ref['loss_points']==120000 and not ref['loss_mismatches'] and not ref['evaluation_mismatches']
> plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
> fig,ax=plt.subplots(figsize=(11,4.4),layout='constrained');left=np.zeros(5)
> for category,label,color in [('success','Reached goal','#007f73'),('wall_cycle','Stuck repeating an invalid action','#b77d47'),('two_state_cycle','Two-state cycle','#8085ba')]:
>  vals=np.array([traj[m]['heldout']['counts'].get(category,0) for m in modes]);ax.barh(english,vals,left=left,label=label,color=color)
>  for i,(v,l) in enumerate(zip(vals,left)):
>   if v>=5:ax.text(l+v/2,i,str(v),ha='center',va='center',color='white',fontsize=10)
>  left+=vals
> ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Number of fixed held-out mazes (100 total)',title='Why greedy evaluation fails at training step 120,000')
> fig.legend(loc='outside lower center',ncol=1,frameon=False);fig.savefig(P/'failure_modes.png',dpi=160);plt.close(fig)
> rows=[]
> for m,name in zip(modes,names):
>  r=s[m];w=r['last20k'];rows.append(f"| {name} | {r['fixed_final']['fixed']['heldout']['success']:.0%} | {w['unique_states_mean']:.1f} | {w['top10_fraction']:.2%} | {w['goals']} |")
> pol=[]
> for m,name in zip(modes,names):
>  a=audit[m]['heldout'];pol.append(f"| {name} | {a['greedy_invalid_fraction']:.2%} | {a['greedy_shortest_fraction']:.2%} | {a['q_mae']:.4f} |")
> text=f'''**ε-greedy + MAB 为什么变差：受控实验报告**
>
> 本次最有证据支持的解释是：用尚未学好的 Q 网络参与采样后，新数据集中在少数重复转移上；同时，MAB 不记录贪心动作的实际反馈，会进一步影响探索效果。冻结整个 MAB 会造成更严重的采样退化。没有证据表明这是 seed 42 或 TensorBoard 渲染错误，也不能把全部差距归结为撞墙、回头路或成功轨迹数量中的单一指标。
>
> 对同一个 ε 调度，只把“网络所选动作也更新 MAB”打开，固定未见迷宫成功率从 **1/100 提高到 11/100**。这支持反馈缺失是一个实际影响因素。全程 MAB 的结果仍为 **26/100**，所以这个改变只恢复了部分差距。以下结论限定于本次 seed 42、5×5 迷宫和前 120000 个训练步，不声称所有 ε-greedy 算法都会变差。
>
> **实验如何控制**
>
> 每组先执行 819199 个 MAB warmup 环境步，再执行 120000 个网络训练步。学习从总环境步 819200 开始，网络更新、target 更新均为每步一次。seed=42，replay 容量 500000，batch=64，gamma=0.99，tau=0.005，训练每个 episode 最多 2500 步，warmup 最多 5000 步。MAB 的 c=2、lr=0.1。
>
> ε(t)=max(0.05, 1−0.95×t/204800)，t 是 warmup 之后的训练步。在 120000 步时，ε≈0.4434。本报告中的 ε-greedy 探索分支使用 MAB；没有把这些结果当作标准 uniform ε-greedy 的结论。
>
> 前三组复制同一个内存中的 warmup 检查点：网络、环境及 RNG 状态、replay 内容及采样 RNG、MAB 的 Q/N 及 RNG。两组干预按相同代码与 seed 重新构建 warmup。每组从相同初始化开始，没有继承另一组已训练的网络。原来每 10000 步的随机评估顺序保留，另用独立环境对固定的前 100 个训练迷宫和前 100 个未见迷宫评估，限 25 步、纯贪心，不更新 MAB。
>
> 全程 MAB 对照的 **120000 个 loss 和 72 个原始评估值均与 v1 精确相同**，见[基准核对]({P}/reference_check.json)。实验读取的是数值，不依赖图表平滑或缓存。
>
> **结果：学习和新数据分布一起变化**
>
> 下表的成功率是第 120000 步的固定 100 个未见迷宫结果。采样统计取训练步 100001–120000：每个窗口含 5000 条新 transition，共四个窗口；“状态数”和“前十占比”是窗口均值，“到达目标”是该 20000 步的总次数。
>
> | 策略 | 固定未见迷宫成功率 | 每 5000 条新数据的不同状态数 | 最常见 10 条 transition 的占比 | 到达目标次数 |
> |---|---:|---:|---:|---:|
> {chr(10).join(rows)}
>
> ![学习曲线]({P}/learning.png)
>
> ![采样分布]({P}/sampling.png)
>
> 全程 MAB 的前十 transition 只占 2.52%，而仅探索动作更新 MAB 的 ε-greedy 达到 44.15%；不同状态数从 489.3 降至 297.0。这里描述的是进入 replay 的新数据，并不是整个 500000 条 replay 内容的去重统计。不同状态按 Tensor 数值识别；完整 transition 的 key 包含 state、action、原始 reward、next_state、terminated、truncated。
>
> 采样集中意味着重复的少数状态动作在更新数据中获得更大权重。结合受控策略替换后的学习差异，这是支持“策略引起数据分布退化”的证据；本实验没有单独操纵 replay 重复权重，因此不把重复率本身宣称为已经独立证明的唯一原因。
>
> **三个具体发现及干预边界**
>
> 1. **冻结 MAB 会破坏 UCB 的持续探索机制。** 已有状态的 Q 和 N 都不变时，UCB 分数也不变；所有动作都尝试过的状态反复选同一个 argmax。尚有 N=0 的动作时，选过之后仍不增加 N，它们始终留在“未尝试”集合。这是代码层面确定的行为。120000 步中，冻结组只采到 101 次到达目标，而持续更新探索动作组为 802 次，全程 MAB 为 947 次。冻结组的状态覆盖也明显更低，但本实验不把冻结解释为另一组持续更新 MAB 失败的原因。
>
> 2. **只记录探索动作，会漏掉真实发生的贪心动作。** 第 100001–120000 步，所有 ε-greedy 组的分支 RNG 一致，都提出 10115 个贪心动作。只更新探索动作的 MAB 对这部分真实反馈不做统计；“所有动作更新”组保持 ε 调度和选动作公式，唯一改动是把贪心动作的反馈也用于 MAB 的状态表、Q 和 N。固定未见成功率从 1% 升到 11%，新数据覆盖从 297.0 升到 436.8，说明这是可修正的影响因素，但并未恢复到全程 MAB 的 26%。
>
> 3. **减少回头路、增加成功采样，并不足以解决问题。** 折返干预只在网络建议立即回到上一状态时调用 MAB 接管，仍允许 MAB 选择回退；它没有禁掉迷宫中必要的回退。最后 20000 步的即时往返比例从 15.00% 降到 5.03%，整段 120000 步采到目标次数反而达到 1016，超过全程 MAB 的 947；然而固定未见成功率只有 3%。因此不能把“少回头”或“采到更多正奖励”直接当作学习一定改善的原因。所有动作更新组的即时往返比例为 17.05%，却比普通 ε-greedy 学得更好，也说明单看这一指标会误判。
>
> **网络最终学到了什么**
>
> 在同样的 100 个未见迷宫中，枚举所有能到达终点的非终点位置，共 1410 个状态，比较网络动作与迷宫 BFS 最短路。以下不是随机抽到的训练状态；所有组在同一批状态上审计。
>
> | 策略 | 贪心选择无效动作的状态比例 | 贪心选择最短路动作的状态比例 | Q 与解析最优 Q 的平均绝对误差 |
> |---|---:|---:|---:|
> {chr(10).join(pol)}
>
> 解析 Q 使用环境的实际奖励：有效非终点 −0.01、无效 −0.05、到达目标 +1，gamma=0.99，以无时间截断的确定性迷宫 MDP 为参照；它不是有限 25 步评估回报的预测值。这里的全局平均 Q 偏差主要为负，因此没有把“max 导致过估计”当作已经证实的根因。
>
> 对每个固定迷宫从起点实际追踪贪心策略，普通 ε-greedy 组有 **92/100** 陷入反复选择无效动作的同一状态，7/100 陷入两状态循环，只有 1/100 成功。相同状态下固定网络给出相同动作，所以这些是确定性的闭环，不是碰巧一次走错。全程 MAB 训练的网络对应为 45、29、26。图中每一行都是相同的 100 个迷宫。
>
> ![评估失败类型]({P}/failure_modes.png)
>
> 较高的状态级动作错误率，可以在完整路径中表现为很低的 episode 成功率；不能把单步最短路选择率直接当作迷宫成功率。
>
> **怎样使用这些结果**
>
> 若继续采用 ε-greedy + MAB，优先把 MAB 拆成“选动作”和“观察 transition”两部分，让所有实际动作更新 MAB；不要冻结 N。这一方向在本次有部分恢复效果。仅仅禁止或回避回头路没有恢复性能，不建议把它当成已验证的修复。
>
> 下一步可以另测更慢的 ε 衰减、训练初期继续由 MAB 主导、或对重复 replay 样本做控制；这些是待验证方案，本次没有把它们包装成已证实解决方案。目前已经确认的差距包括 MAB 反馈缺失和明显的采样分布变化，仍未完全分离“哪些数据权重变化造成了剩余 26% 对 11% 的差距”。
>
> 本次只有 seed 42，也只跑到 120000 个训练步，没有验证原来约 410000 步的最终排名。固定 100 个迷宫只是指定子集，因此保留了原协议的随机评估作为补充：第 120000 步，全程 MAB / 普通 ε-greedy / 冻结 / 折返干预 / 全动作更新的未见迷宫成功率分别为 28%、0%、8%、2%、12%。这两种评估的具体值有差别，但没有改变本次主要判断。
>
> **复查材料**
>
> [实验设置]({P}/manifest.json)、[汇总数据]({P}/summary.json)、[固定状态审计]({P}/policy_audit.json)、[逐迷宫失败分类]({P}/trajectory_audit.json)、[源码哈希]({P}/source_hashes.json)。每组另有 `*_loss.npy`、`*_windows.json`、`*_eval.json`、`*_latest_weights.pt` 和 `*_done.json`。绘图输入直接来自这些文件，没有平滑 success 曲线。
>
> [主实验脚本]({P}/experiment.py)、[折返干预]({P}/intervention.py)、[全动作更新干预]({P}/observe_all.py)、[绘图脚本]({P}/summarize.py)。脚本从独立的 `frozen_project` 源码快照导入；项目训练文件和原始 v1/v2 日志没有被修改。
> '''
> (P/'report.md').write_text(text)
> (P/'audit_complete.json').write_text(json.dumps({'modes':modes,'training_steps_each':120000,'finite_loss_points_each':120000,'fixed_eval_outcomes_match_saved_metrics':True,'baseline_loss_matches_v1':120000,'baseline_evaluation_values_match_v1':72,'source_files':len(json.loads((P/'source_hashes.json').read_text()))},indent=2))
> print('Report and audit written:',P/'report.md')
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/report.py``
>
> </details>
>
> Viewed 3 images
>
> <details><summary>Ran a command</summary>
>
> - Ran `cat analysis/epsilon_mab_20260908/audit_complete.json
>  git -C project/IncompleteBaseline diff --stat`
>
> Tool call
>
> Tool: open_in_codex
> Status: completed
>
> </details>
>
> System error
>
> You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 12:04 PM.
</details>

> 继续工作

<details><summary>21 previous messages</summary>

> 我会继续追查剩余差距。上一轮中，让 MAB 也学习贪心动作的反馈，成功率从 1% 提高到了 11%，但全程 MAB 是 26%。这轮会直接检验“重复 transition 在 replay 中占据过大权重”这个解释：保持 ε-greedy 策略不变，只改变 replay 的采样权重，观察学习是否恢复。
>
> <details><summary>Explored 4 files, a search, a list, ran 3 commands</summary>
>
> <details><summary>Explored 4 files, a search, a list</summary>
>
> - Read `./analysis/epsilon_mab_20260908/audit_complete.json`
> - Read `./analysis/epsilon_mab_20260908/manifest.json`
> - Read `./analysis/epsilon_mab_20260908/experiment.py`
> - Searched for `TensorBoard merges|controlled comparisons|full transition` in `MEMORY.md`
> - Read `./analysis/epsilon_mab_20260908/frozen_project/IncompleteBaseline/algorithms/common/replay_buffer.py`
> - Listed files in `analysis`
>
> </details>
>
> - Ran `cat > analysis/epsilon_mab_20260908/replay_weighting.py <<'PY'
> """Keep the original FIFO contents; change only sampling over occurrences vs values."""
> import copy,collections
> import torch
> from IncompleteBaseline.algorithms.common.replay_buffer import ReplayBuffer,TransitionBatch
>
> def transition_key(t):
>  def enc(s):
>   x=s.detach().cpu().contiguous()
>   return tuple(x.shape),str(x.dtype),x.numpy().tobytes()
>  return (*enc(t.state),t.action,float(t.reward),*enc(t.next_state),t.terminated,t.truncated)
>
> class IndexedReplay(ReplayBuffer):
>  def __init__(self,source,unique=False):
>   super().__init__(source.buffer_size,source.seed,source.input_dim)
>   self.unique=unique;self._queue=collections.deque();self._active=[];self._positions={};self._counts={};self._reps={};self._sample_counts=collections.Counter();self._sample_total=0;self._sample_positive=0;self._sample_selfloop=0
>   for t in source.pool:self.push(t)
>   self._rng=copy.deepcopy(source._rng)
>  def push(self,t):
>   k=transition_key(t)
>   if len(self._queue)==self.buffer_size:
>    old=self._queue.popleft();self._counts[old]-=1
>    if not self._counts[old]:
>     idx=self._positions.pop(old);last=self._active.pop()
>     if idx<len(self._active):self._active[idx]=last;self._positions[last]=idx
>     del self._counts[old];del self._reps[old]
>   if k in self._counts:
>    k=self._active[self._positions[k]];self._counts[k]+=1
>   else:
>    self._positions[k]=len(self._active);self._active.append(k);self._counts[k]=1;self._reps[k]=t
>   self._queue.append(k);super().push(t)
>  def sample(self,batch_size):
>   keys=self._rng.sample(self._active if self.unique else self._queue,batch_size)
>   batch=[self._reps[k] for k in keys]
>   self._sample_counts.update(keys);self._sample_total+=len(keys)
>   self._sample_positive+=sum(k[4]>0 for k in keys)
>   self._sample_selfloop+=sum(k[:3]==k[5:8] for k in keys)
>   return TransitionBatch(
>    states=torch.concat([torch.as_tensor(t.state,dtype=torch.float32) for t in batch],dim=0),
>    actions=torch.stack([torch.as_tensor(t.action,dtype=torch.long).reshape(1) for t in batch]),
>    rewards=torch.stack([torch.as_tensor(t.reward,dtype=torch.float32).reshape(1) for t in batch]),
>    next_states=torch.concat([torch.as_tensor(t.next_state,dtype=torch.float32) for t in batch],dim=0),
>    terminated=torch.stack([torch.as_tensor(t.terminated,dtype=torch.bool).reshape(1) for t in batch]),
>    truncated=torch.stack([torch.as_tensor(t.truncated,dtype=torch.bool).reshape(1) for t in batch]))
>  def diagnostics(self):
>   total=self._sample_total
>   r={'buffer_entries':len(self.pool),'buffer_unique_transitions':len(self._active),'buffer_top10_occurrence_fraction':sum(sorted(self._counts.values(),reverse=True)[:10])/len(self.pool),'sampled_entries':total,'sampled_unique_transitions':len(self._sample_counts),'sampled_top10_fraction':sum(x for _,x in self._sample_counts.most_common(10))/total,'sampled_positive_fraction':self._sample_positive/total,'sampled_selfloop_fraction':self._sample_selfloop/total}
>   self._sample_counts.clear();self._sample_total=self._sample_positive=self._sample_selfloop=0
>   return r
> PY
> cat > analysis/epsilon_mab_20260908/check_replay_weighting.py <<'PY'
> import sys,random,collections
> from pathlib import Path
> import torch
> sys.path.insert(0,str(Path(__file__).resolve().parent/'frozen_project'))
> from IncompleteBaseline.algorithms.common.replay_buffer import ReplayBuffer,Transition
> from replay_weighting import IndexedReplay,transition_key
>
> def trans(i,trunc=False):
>  return Transition(torch.tensor([[float(i),0.]]),i%4,float(i%2),torch.tensor([[float(i+1),0.]]),False,trunc)
> a=ReplayBuffer(23,42,2)
> for i in range(50):a.push(trans(i%7))
> b=IndexedReplay(a);u=IndexedReplay(a,unique=True)
> r=random.Random(88)
> for i in range(150):
>  t=trans(r.randrange(9),bool(i%5==0));a.push(t);b.push(t);u.push(t)
>  actual=collections.Counter(transition_key(t) for t in a.pool)
>  for obj in [b,u]:
>   assert actual==obj._counts
>   assert list(obj._queue)==[transition_key(t) for t in a.pool]
>   assert len(obj._active)==len(set(obj._active))==len(obj._reps)==len(obj._positions)
>   assert all(obj._positions[k]==idx for idx,k in enumerate(obj._active))
>  ba=a.sample(5);bb=b.sample(5)
>  for field in ['states','actions','rewards','next_states','terminated','truncated']:assert torch.equal(getattr(ba,field),getattr(bb,field))
>  bu=u.sample(5)
>  keys=[(tuple(bu.states[j].tolist()),int(bu.actions[j]),float(bu.rewards[j]),tuple(bu.next_states[j].tolist()),bool(bu.terminated[j]),bool(bu.truncated[j])) for j in range(5)]
>  assert len(set(keys))==5
> assert transition_key(trans(1))!=transition_key(trans(1,True))
> print('PASS: 150 FIFO eviction/duplicate cases; occurrence batches exactly match; unique batches have no duplicate values; truncation preserved.')
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/check_replay_weighting.py`
>
> - Ran `python3 - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908');s=(p/'experiment.py').read_text()
> s=s.replace("default=['all_mab','eps_live','eps_frozen']","default=['eps_occurrence_control','eps_unique_replay']").replace("OUT/'manifest.json'","OUT/'replay_weighting_manifest.json'")
> s=s.replace("from IncompleteBaseline.algorithms.common.replay_buffer import Transition","from IncompleteBaseline.algorithms.common.replay_buffer import Transition\nfrom replay_weighting import IndexedReplay")
> s=s.replace("s=state.clone();prev=None;", "a.replaybuffer=IndexedReplay(a.replaybuffer,unique=(mode=='eps_unique_replay'))\n steps=20000 if mode=='eps_occurrence_control' else args.steps\n s=state.clone();prev=None;")
> s=s.replace("range(1,args.steps+1)","range(1,steps+1)").replace("{'steps':args.steps,'elapsed'","{'steps':steps,'elapsed'")
> s=s.replace("windows.append(rec);", "rec['replay_sampling']=a.replaybuffer.diagnostics()\n   windows.append(rec);")
> s=s.replace("print('DONE',mode,flush=True)","""if mode=='eps_occurrence_control':
>   expected=np.load(OUT/'eps_live_loss.npy')[:steps]
>   actual=np.asarray(losses,dtype=np.float32)
>   differences=np.flatnonzero(expected!=actual)
>   proof={'loss_points':steps,'mismatch_count':int(len(differences)),'first_mismatch':int(differences[0]+1) if len(differences) else None}
>   (OUT/'occurrence_control_check.json').write_text(json.dumps(proof,indent=2));print('CONTROL CHECK',proof,flush=True)
>   assert not len(differences)
>  print('DONE',mode,flush=True)""")
> compile(s,'replay_experiment.py','exec');(p/'replay_experiment.py').write_text(s)
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B -u analysis/epsilon_mab_20260908/replay_experiment.py > analysis/epsilon_mab_20260908/replay_weighting.log 2>&1`
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B - <<'PY'
> import sys,json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908').resolve();sys.path[:0]=[str(p),str(p/'frozen_project')]
> from IncompleteBaseline.algorithms.common.replay_buffer import ReplayBuffer,Transition
> from replay_weighting import IndexedReplay
> import torch
> r=ReplayBuffer(100,42,2)
> for i in range(100):
>  x=0 if i<90 else 1
>  r.push(Transition(torch.tensor([[float(x),0.]]),0,0.,torch.tensor([[float(x),1.]]),False,False))
> out={}
> for unique in [False,True]:
>  b=IndexedReplay(r,unique);n=sum(int(b.sample(1).states[0,0].item()==0) for _ in range(10000));out[str(unique)]=n/10000
> assert abs(out['False']-.9)<.02 and abs(out['True']-.5)<.02
> (p/'sampler_probability_check.json').write_text(json.dumps(out,indent=2));print(out)
> PY
> tail -3 analysis/epsilon_mab_20260908/replay_weighting.log`
>
> </details>
>
> 新的 sampler 已通过检查：保留重复权重时，抽到的 batch 与原实现逐项一致；改为按不同 transition 均匀抽样后，重复条目不会再获得额外权重。正式训练前还会跑一个 20000 步对照，确认增加索引和诊断本身没有改变原来的训练结果。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `tail -3 analysis/epsilon_mab_20260908/replay_weighting.log`
> - Ran `python3 - <<'PY'
> import json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> a=json.loads((p/'eps_occurrence_control_windows.json').read_text())
> print(json.dumps(a[-1]['replay_sampling'],indent=2))
> PY`
> - Ran `tail -4 analysis/epsilon_mab_20260908/replay_weighting.log`
> - Ran `cat analysis/epsilon_mab_20260908/occurrence_control_check.json
> python3 - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> s=(p/'policy_audit.py').read_text().replace("mode=file.name.replace('_latest_weights.pt','');q=", "mode=file.name.replace('_latest_weights.pt','')\n if mode not in ['all_mab','eps_live','eps_unique_replay']:continue\n q=").replace("P/'policy_audit.json'","P/'replay_policy_audit.json'")
> compile(s,'replay_policy_audit.py','exec');(p/'replay_policy_audit.py').write_text(s)
> s=(p/'trajectory_audit.py').read_text().replace("mode=f.name.replace('_done.json','');q=", "mode=f.name.replace('_done.json','')\n if mode not in ['all_mab','eps_live','eps_unique_replay']:continue\n q=").replace("P/'trajectory_audit.json'","P/'replay_trajectory_audit.json'")
> compile(s,'replay_trajectory_audit.py','exec');(p/'replay_trajectory_audit.py').write_text(s)
> PY`
>
> </details>
>
> 20000 步对照通过：loss 与上一轮普通 ε-greedy 逐点完全一致。等权采样组已开始，前 10000 步固定未见迷宫成功率是 6%，原组同期是 3%；这个差距还小，我会跑到相同的 120000 步再判断。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `python3 - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908');s=(p/'replay_experiment.py').read_text()
> s=s.replace("default=['eps_occurrence_control','eps_unique_replay']","default=['all_mab_unique_replay']").replace("OUT/'replay_weighting_manifest.json'","OUT/'all_mab_unique_manifest.json'")
> s=s.replace("unique=(mode=='eps_unique_replay')","unique=(mode in ['eps_unique_replay','all_mab_unique_replay'])").replace("mode=='all_mab' else", "mode in ['all_mab','all_mab_unique_replay'] else")
> compile(s,'all_mab_unique_experiment.py','exec');(p/'all_mab_unique_experiment.py').write_text(s)
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B -u analysis/epsilon_mab_20260908/all_mab_unique_experiment.py > analysis/epsilon_mab_20260908/all_mab_unique.log 2>&1
> tail -2 analysis/epsilon_mab_20260908/replay_weighting.log`
> - Ran `tail -2 analysis/epsilon_mab_20260908/replay_weighting.log
> tail -2 analysis/epsilon_mab_20260908/all_mab_unique.log`
>
> </details>
>
> 现在出现了明显恢复：按不同 transition 等权采样的 ε-greedy 组，在 60000 步时固定未见迷宫成功率达到 45%；原来的 ε-greedy 同期是 1%。我补了一组“全程 MAB＋等权 replay”，用来区分这是普遍的 replay 改进，还是专门消除了 ε-greedy 的劣势。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `python3 - <<'PY'
> import json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> for m in ['eps_occurrence_control','eps_unique_replay']:
>  w=[r for r in json.loads((p/f'{m}_windows.json').read_text()) if r['step']<=20000]
>  print(m)
>  for k in ['sampled_unique_transitions','sampled_positive_fraction','sampled_selfloop_fraction','sampled_top10_fraction','buffer_unique_transitions']:
>   print(k,sum(r['replay_sampling'][k] for r in w)/len(w))
> PY`
> - Ran `tail -2 analysis/epsilon_mab_20260908/replay_weighting.log
> tail -2 analysis/epsilon_mab_20260908/all_mab_unique.log`
>
> </details>
>
> 等权采样确实改变了网络“学到哪些数据”：前 20000 步中，batch 里的正奖励 transition 比例从约 0.91% 提高到 2.17%，每 5000 次更新实际覆盖的不同 transition 也从约 12.2 万增加到 15.7 万。值得注意的是，batch 中原地不动的比例没有下降，所以收益不能简单解释成“少学了撞墙样本”。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `cat > analysis/epsilon_mab_20260908/replay_followup_plot.py <<'PY'
> from pathlib import Path
> import json,os
> import numpy as np
> os.environ.setdefault('MPLCONFIGDIR','/tmp/mab-mpl-cache')
> import matplotlib
> matplotlib.use('Agg')
> import matplotlib.pyplot as plt
> P=Path(__file__).resolve().parent
> modes=['all_mab','eps_live','all_mab_unique_replay','eps_unique_replay']
> labels=['Full MAB / occurrence-weighted','epsilon-greedy / occurrence-weighted','Full MAB / equal transition weights','epsilon-greedy / equal transition weights']
> colors=['#007f73','#d56b25','#007f73','#d56b25'];styles=['-','-','--','--']
> plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
> fig,axs=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
> for m,l,c,st in zip(modes,labels,colors,styles):
>  e=json.loads((P/f'{m}_eval.json').read_text());x=np.array([r['step'] for r in e])/1000
>  for ax,split in zip(axs,['train','heldout']):ax.plot(x,[r['fixed'][split]['success'] for r in e],label=l,color=c,ls=st,lw=2.4)
> for ax,title in zip(axs,['Fixed 100 training mazes','Fixed 100 held-out mazes']):ax.set(title=title,xlabel='Training steps (thousands)',ylabel='Greedy success rate (25-step limit)',ylim=(0,1))
> fig.suptitle('Replay weighting changes the result — seed 42',fontsize=16)
> fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
> fig.savefig(P/'replay_weighting_learning.png',dpi=170);fig.savefig(P/'replay_weighting_learning.pdf');plt.close(fig)
> fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
> for m,l,c,st in zip(modes,labels,colors,styles):
>  w=json.loads((P/f'{m}_windows.json').read_text());x=np.array([r['step'] for r in w])/1000
>  axs[0].plot(x,[r['unique_states'] for r in w],label=l,color=c,ls=st,lw=2)
>  axs[1].plot(x,[r['top10_fraction'] for r in w],label=l,color=c,ls=st,lw=2)
> axs[0].set(title='Different states per 5,000 new transitions',xlabel='Training steps (thousands)')
> axs[1].set(title='Top 10 transitions: fraction of new samples',xlabel='Training steps (thousands)',ylim=(0,1))
> fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
> fig.savefig(P/'replay_weighting_sampling.png',dpi=170);plt.close(fig)
> summary={}
> for m in modes:
>  w=json.loads((P/f'{m}_windows.json').read_text());e=json.loads((P/f'{m}_eval.json').read_text());z=w[-4:]
>  summary[m]={'steps':w[-1]['step'],'fixed':e[-1]['fixed'],'official':e[-1]['official'],'total_solved':e[-1]['total_solved'],'last20k':{'unique_states_mean':float(np.mean([r['unique_states'] for r in z])),'top10_fraction':float(np.mean([r['top10_fraction'] for r in z])),'goals':sum(r['solved'] for r in z),'selfloop_fraction':sum(r['selfloop'] for r in z)/sum(r['steps'] for r in z)}}
> (P/'replay_followup_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
> PY
> tail -3 analysis/epsilon_mab_20260908/replay_weighting.log
> tail -2 analysis/epsilon_mab_20260908/all_mab_unique.log`
> - Ran `python3 - <<'PY'
> import json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> print(json.dumps(json.loads((p/'eps_unique_replay_eval.json').read_text())[-1],indent=2))
> PY`
>
> </details>
>
> ε-greedy 的等权 replay 组已完成 120000 步：固定未见迷宫成功率从原来的 1% 提高到 **71%**，原协议的随机评估也从 0% 提高到 **86%**。ε 调度、网络和优化器都没改。这是目前比“漏更新 MAB”更强的证据：原来的 replay 频次权重是关键影响因素。全程 MAB 的等权对照还在完成中。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `sed -n '240,292p' project/IncompleteBaseline/algorithms/dqn/dqn_agent.py
> git -C project/IncompleteBaseline status --short
> tail -2 analysis/epsilon_mab_20260908/all_mab_unique.log
> python3 - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> for name in ['replay_policy_audit.py','replay_trajectory_audit.py']:
>  f=p/name;s=f.read_text().replace("['all_mab','eps_live','eps_unique_replay']","['all_mab','eps_live','eps_unique_replay','all_mab_unique_replay']");f.write_text(s)
> PY`
> - Ran `tail -2 analysis/epsilon_mab_20260908/all_mab_unique.log`
>
> </details>
>
> “全程 MAB＋等权 replay”也明显改善了，100000 步时固定未见迷宫成功率达到 61%。这说明改进作用于 replay 的学习权重，并不是 ε-greedy 专属的小修补。两种行为策略都采用等权 replay 后，原先很大的学习差距正在缩小。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `tail -4 analysis/epsilon_mab_20260908/all_mab_unique.log`
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/replay_followup_plot.py > analysis/epsilon_mab_20260908/replay_followup_summary_stdout.json
> /Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/replay_policy_audit.py > analysis/epsilon_mab_20260908/replay_policy_stdout.json
> /Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/replay_trajectory_audit.py > analysis/epsilon_mab_20260908/replay_trajectory_stdout.json`
>
> </details>
>
> 两组等权对照都完成了：在同一批 100 个未见迷宫上，全程 MAB 是 **72%**，ε-greedy 是 **71%**；原权重下分别是 **26%** 和 **1%**。原来的 25 个百分点差距缩到了 1 个百分点。这个结果支持：在当前配置里，replay 按重复次数分配训练权重，是造成 ε-greedy 学不好的关键因素。
>
> <details><summary>Explored 2 files, ran a command</summary>
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./analysis/epsilon_mab_20260908/replay_followup_summary.json`
> - Read `./analysis/epsilon_mab_20260908/replay_policy_audit.json`
>
> </details>
>
> - Ran `cat > analysis/epsilon_mab_20260908/replay_followup_report.py <<'PY'
> from pathlib import Path
> import json,hashlib
> import numpy as np
> P=Path(__file__).resolve().parent
> s=json.loads((P/'replay_followup_summary.json').read_text());q=json.loads((P/'replay_policy_audit.json').read_text());trace=json.loads((P/'replay_trajectory_audit.json').read_text())
> control=json.loads((P/'occurrence_control_check.json').read_text());assert control['loss_points']==20000 and control['mismatch_count']==0
> modes=['all_mab','eps_live','all_mab_unique_replay','eps_unique_replay']
> for mode in modes:
>  assert s[mode]['steps']==120000 and len(np.load(P/f'{mode}_loss.npy'))==120000
>  assert np.isfinite(np.load(P/f'{mode}_loss.npy')).all()
>  assert json.loads((P/f'{mode}_done.json').read_text())['steps']==120000
>  assert sum(trace[mode]['heldout']['counts'].values())==100
>  assert trace[mode]['heldout']['counts'].get('success',0)==round(s[mode]['fixed']['heldout']['success']*100)
> a=json.loads((P/'all_mab_windows.json').read_text());b=json.loads((P/'all_mab_unique_replay_windows.json').read_text())
> assert len(a)==len(b)==24
> bad=[]
> for x,y in zip(a,b):
>  for k in x:
>   if k not in ['loss_mean','replay_sampling'] and x[k]!=y[k]:bad.append([x['step'],k,x[k],y[k]])
> assert not bad
> samples={}
> for mode in ['eps_occurrence_control','eps_unique_replay']:
>  rows=[r for r in json.loads((P/f'{mode}_windows.json').read_text()) if r['step']<=20000]
>  samples[mode]={k:float(np.mean([r['replay_sampling'][k] for r in rows])) for k in ['sampled_unique_transitions','sampled_positive_fraction','sampled_selfloop_fraction','sampled_top10_fraction']}
> for mode in ['eps_occurrence_control','eps_unique_replay','all_mab_unique_replay']:
>  for r in json.loads((P/f'{mode}_windows.json').read_text()):
>   d=r['replay_sampling'];assert d['buffer_entries']==500000 and 64<=d['buffer_unique_transitions']<=500000 and d['sampled_entries']==320000
> proof={'control':control,'full_mab_behavior_windows_compared':24,'full_mab_behavior_window_mismatches':bad,'main_training_steps_each':120000,'matched_fixed_heldout_mazes':100,'early_batch_statistics':samples,'source_hashes':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['replay_weighting.py','replay_experiment.py','all_mab_unique_experiment.py','check_replay_weighting.py']}}
> (P/'replay_followup_audit.json').write_text(json.dumps(proof,indent=2))
> doc=f'''**2026-09-09 续查：replay 重复次数带来的训练权重是关键影响因素**
>
> 这次做了“行为策略 × replay 权重”的 2×2 对照。只改变 mini-batch 的采样权重，ε-greedy 的学习效果恢复了：固定未见迷宫成功率从 **1% 提高到 71%**。全程 MAB 则从 **26% 提高到 72%**。原来的组间差距是 25 个百分点，等权 replay 下只有 1 个百分点。本次结果不支持“ε-greedy 本身必然学不好”。
>
> | Replay 抽样方式 | 全程 MAB | ε-greedy + 持续更新的 MAB |
> |---|---:|---:|
> | 原实现：均匀抽取 FIFO 中的条目 | 26/100 | 1/100 |
> | 干预：对不同完整 transition 等权抽样 | 72/100 | 71/100 |
>
> 两种策略均使用 seed 42、相同 warmup 协议（819199 步）、120000 个网络训练步、相同网络与优化器、epsilon 调度、环境和奖励。每组固定评估相同的 100 个未见迷宫，25 步上限、纯贪心。ε 从 1.0 线性衰减，在 120000 步为 0.443359375；探索分支使用 MAB，MAB 在探索动作后更新。这里没有同时改成“贪心动作也更新 MAB”。
>
> ![2×2 学习曲线]({P}/replay_weighting_learning.png)
>
> 原协议的随机评估也支持恢复：120000 步时，全程 MAB 的成功率为 28%→74%，ε-greedy 为 0%→86%。固定子集与随机子集的具体数值不同，因此两组均报告，不选择更好看的一个作为唯一结果。
>
> **到底改了什么**
>
> 原实现均匀抽取 FIFO 中的每条记录。同一个 transition 如果有 nᵢ 份，抽到它的总概率就是 nᵢ/N，其中 N=500000。这是按出现次数给训练目标加权，并不是随机种子出错。
>
> 新实现继续保留相同容量、相同先进先出规则的 500000 条记录，只额外维护当前 FIFO 中各个不同 transition 的索引。每个不同 transition 的抽样概率为 1/U，U 是当前 FIFO 内不同 transition 的数量；batch 内按这些不同值无放回抽样。过期且没有剩余副本的 transition 会从索引删除，没有无限保留历史数据。
>
> 去重 key 包含 state 的数值、形状、dtype，action，原始 reward，next_state 的数值、形状、dtype，terminated 和 truncated。没有只按 state 去重，也没有把 Tensor 对象身份当作内容。没有改变 reward、TD target 公式、网络结构、学习率、ε 调度或 MAB 选择公式。
>
> **为什么这个对照比“新数据重复率很高”更有说服力**
>
> 先实现一个保留原重复权重的索引 sampler，对它做了 20000 个训练步的回归对照：loss 与上一轮普通 ε-greedy **逐点完全一致**。另外通过了 150 轮重复插入、FIFO 淘汰及 batch 对比测试，验证 truncated 不被合并。人工构造 90 份 A、10 份 B，采样 10000 次时，A 的占比从原权重的 89.93% 变为等权的 49.84%，与预期一致。
>
> 全程 MAB 的行为不依赖 Q 网络，两种 replay 权重下的数据采集按构造相同：120000 步都到达目标 947 次；24 个采样窗口中，所有已记录的行为统计完全一致。但评估从 26% 变成 72%。这把“采集到了不同轨迹”与“相同数据被如何用于学习”区分开了，支持 replay 权重本身是关键影响因素。
>
> ε-greedy 则会把网络学到的策略反馈到后续采样。等权 replay 让网络先学得更好，后续数据也改善了。最后 20000 步中，每 5000 条新数据的不同状态数从 **297.0 增加到 1643.5**；最常见 10 条 transition 的占比从 **44.15% 降到 13.84%**；这段期间到达目标次数从 **96 增加到 596**。这里描述的是新收集的数据，不能与整个 replay 的占比或实际 batch 占比混淆。
>
> ![新采样数据变化]({P}/replay_weighting_sampling.png)
>
> 全程 MAB 的两条采样曲线重合是预期结果：它们的行为相同，差别发生在 replay 采样和网络更新中。
>
> **实际训练 batch 的变化**
>
> 下面比较前 20000 个训练步，将其分成四个 5000 步窗口。每个窗口有 5000×64=320000 次样本使用。
>
> | 指标（窗口均值） | 原频次权重 | 不同 transition 等权 |
> |---|---:|---:|
> | 被用于训练的不同 transition 数 | {samples['eps_occurrence_control']['sampled_unique_transitions']:,.0f} | {samples['eps_unique_replay']['sampled_unique_transitions']:,.0f} |
> | 正奖励 transition 占比 | {samples['eps_occurrence_control']['sampled_positive_fraction']:.3%} | {samples['eps_unique_replay']['sampled_positive_fraction']:.3%} |
> | 原地不动 transition 占比 | {samples['eps_occurrence_control']['sampled_selfloop_fraction']:.2%} | {samples['eps_unique_replay']['sampled_selfloop_fraction']:.2%} |
>
> 正奖励（在此环境对应到达终点）的样本比例从约 0.91% 增加到 2.17%，不同 transition 的覆盖也提高了。原地不动样本比例反而略增，因此不能简单说“删掉撞墙样本所以变好”。这个干预同时改变了稀有终点信号和其他状态动作的相对权重；本次没有进一步单独分离两者各自的贡献。
>
> 同一批 1410 个可到达终点的未见迷宫状态上，ε-greedy 网络选择无效动作的比例从 **27.94% 降到 0.64%**，选择最短路动作的比例从 **62.70% 升到 95.04%**。Q 相对解析最优 Q 的平均绝对误差从 **0.1761 降到 0.0195**。解析 Q 的定义沿用[上一轮报告]({P}/report.md)：使用实际奖励与 gamma=0.99 的确定性迷宫、无时间截断参照，不是有限 25 步评估回报。
>
> **结论与范围**
>
> 当前证据支持：原 replay 的重复频次权重限制了 Q 学习；未学好的网络参与 ε-greedy 采样后，又会把后续数据集中到少数转移上。对 replay 做等权干预，既提高了全程 MAB 在相同采集数据上的学习效果，也基本消除了本次 ε-greedy 相对全程 MAB 的差距。这比上一轮仅观察到采样集中、或只更新 MAB 反馈的证据更直接。
>
> 这不表示均匀抽 FIFO 条目是代码错误，而是这个数据分布下的一项不利学习权重。也不能据此声称“所有重复数据都有害”或“所有环境都应该按完整 transition 去重”。本次是确定性 5×5 迷宫、seed 42、120000 个训练步；没有验证多个 seed、原计划约 410000 步的最终排名、随机转移环境或其他任务。
>
> 若继续改实现，优先把“按不同完整 transition 等权采样”作为可选 replay sampler，与原方式保留对照。这一方向现在有直接实验支持。项目训练代码没有被修改，本次新增的实现、数据和图表全部放在分析目录中。
>
> [数值汇总]({P}/replay_followup_summary.json)、[验证记录]({P}/replay_followup_audit.json)、[状态级 Q 审计]({P}/replay_policy_audit.json)、[逐迷宫轨迹审计]({P}/replay_trajectory_audit.json)。
>
> [Sampler 实现]({P}/replay_weighting.py)、[ε-greedy 实验脚本]({P}/replay_experiment.py)、[全程 MAB 对照脚本]({P}/all_mab_unique_experiment.py)、[绘图脚本]({P}/replay_followup_plot.py)。训练脚本从保存的 frozen_project 源码快照导入，避免当前工作区的后续修改影响这组对照。
> '''
> (P/'replay_followup_report.md').write_text(doc)
> old=P/'report.md';text=old.read_text();notice=f'**后续更新：** [2026-09-09 replay 权重对照]({P}/replay_followup_report.md) 已完成，补上了本文当时尚未做的 replay 权重干预。\n\n'
> if notice not in text:old.write_text(notice+text)
> print('Follow-up report and checks passed.')
> PY
> /Users/jay/miniconda3/envs/rl/bin/python -B analysis/epsilon_mab_20260908/replay_followup_report.py`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> <details><summary>Explored a file</summary>
>
> - Read `./analysis/epsilon_mab_20260908/replay_followup_audit.json`
>
> </details>
>
> Viewed 2 images
>
> 验证已完成，图表也已检查。最有力的对照是：全程 MAB 两组采集数据的 24 个窗口统计完全一致，仅改变 replay 的采样权重，成功率就从 26% 提高到 72%。这支持 replay 中重复经验的权重是关键影响因素。
>
> <details><summary>Explored 2 files</summary>
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./analysis/epsilon_mab_20260908/replay_followup_report.md`
> - Read `~/.codex/memories/MEMORY.md`
>
> </details>
>
> Tool call
>
> Tool: open_in_codex
> Status: completed
>
> </details>
</details>

继续验证后，发现了一个关键因素：**replay 中重复 transition 的出现次数，会影响 Q 网络的学习效果。**

保持 seed 42、warmup、网络参数和探索策略不变，只改变 replay 抽样权重。训练 120,000 步后，在相同的 100 个未见迷宫上：

| Replay 抽样方式 | 全程 MAB | ε-greedy + MAB |
|---|---:|---:|
| 原方式：均匀抽取每条记录 | 26% | 1% |
| 对不同完整 transition 等权抽样 | **72%** | **71%** |

原方式下，同一个 transition 出现 100 次，其总抽样概率就是出现一次的 transition 的 100 倍。新方式保留原来的 FIFO buffer，只调整抽样权重。

![训练对照曲线](./analysis/epsilon_mab_20260908/replay_weighting_learning.png)

这次有两个重要验证：

- **20,000 步回归对照通过**：保留原权重的测试 sampler，loss 与之前逐点完全一致。
- **全程 MAB 对照**：两组采集数据的 24 个窗口统计完全一致，仅改变 replay 权重，成功率就从 26% 提高到 72%。

因此，目前证据支持：重复经验的权重限制了 Q 学习，而不成熟的 Q 又通过 ε-greedy 让后续采样更加集中。调整 replay 权重后，这次实验中的性能差距从 **25 个百分点缩小到 1 个百分点**。

结论目前限于 **seed 42、120,000 步、这个确定性迷宫环境**。项目训练代码没有修改；实验实现、数据和图表已整理到[完整报告](./analysis/epsilon_mab_20260908/replay_followup_report.md)。

> 你是指修改它的采样概率吗？就是每个 transition 的采样概率为它在这个 replay buffer 里面的出现频率的倒数？
>
> 你是如何想到这一点的？说出背后的思维

对，你理解的方向是对的，但要区分 **buffer 里的每条记录**和**内容相同的一类 transition**。

假设 buffer 有 100 条记录：

- transition A 重复了 90 次。
- transition B 重复了 10 次。

两种抽样方式的区别是：

| 抽样方式 | 抽到 A 的概率 | 抽到 B 的概率 |
|---|---:|---:|
| 原来的均匀抽记录 | 90% | 10% |
| 实验中的不同 transition 等权 | 50% | 50% |

**如果对每条记录按其重复次数的倒数加权，就能得到这个单次抽样效果。** 假设一种 transition 出现了 \(c_i\) 次，buffer 中共有 \(U\) 种不同的 transition，那么归一化后：

\[
P(\text{某一条记录})=\frac{1}{U c_i}
\]

把它的所有副本加起来：

\[
P(\text{这种 transition})=c_i\frac{1}{Uc_i}=\frac1U
\]

不过，我实际实现的是：**维护当前 buffer 中不同完整 transition 的索引，从这些不同值中均匀、无放回地抽取 batch。** 因此同一个 batch 不会出现内容完全相同的 transition。单纯给原始记录加逆频率权重，不一定能保证这一点。

这里的“相同”要求整个 `(state, action, reward, next_state, terminated, truncated)` 相同，不能只看 state。

我提出这个实验，依据的是下面这条可检验的分析路径。

**首先，之前的数据暴露了“训练样本集中”的现象。** ε-greedy 组最后 20,000 步的新数据中，最常见的 10 条 transition 占了 44.15%。这提示网络可能反复训练少量经验，但这个现象本身不能证明它是原因：也可能是网络没学好，才反复走同样的路。

**其次，均匀抽记录，其实隐含了按出现次数加权。** 普通 replay 对应的平均损失可以写成：

\[
L_{\text{原方式}}=\sum_{i=1}^{U}\frac{c_i}{N}\,\ell_i
\]

所以，同一条经验出现得越多，它对优化目标的影响越大。“记录抽样公平”并不等于“不同经验的训练权重相等”。在这个确定性迷宫里，重复撞同一面墙，会增加这条经验的权重，却没有增加新的转移信息。

**因此，实验要回答的是：已有经验足不足以学好，只是训练权重不合适？** 把目标改成不同 transition 等权：

\[
L_{\text{等权}}=\frac1U\sum_{i=1}^{U}\ell_i
\]

尤其是全程 MAB 组，动作不依赖 Q 网络，可以保持采集过程相同，单独检查训练权重的影响。它从 26% 提高到 72%，是这次最有力的证据。

这仍然不能证明“重复经验都应该降权”。等权抽样同时提高了稀有终点经验和其他少见状态动作的权重；**究竟哪一部分贡献最大，目前还没有单独分离验证。**

> 那我该如何修改我的那个 sample replay buffer 里面的 sample 方法？有没有简单一点的？

<details><summary>2 previous messages</summary>

> 我先看一下你现在的 replay buffer 实现，再给你一个尽量只改 `sample()` 的简洁版本。
>
> <details><summary>Explored a file, a search</summary>
>
> - Searched for `class ReplayBuffer|def sample|class Transition` in `replay_buffer.py`
> - Read `./project/IncompleteBaseline/algorithms/common/replay_buffer.py`
>
> </details>
</details>

看了你的 [replay_buffer.py](./project/IncompleteBaseline/algorithms/common/replay_buffer.py:64)，**最简单的写法就是：先按内容去重，再均匀抽样。** 不需要显式计算逆频率。

把 `sample()` 开头这一句：

```python
batch = self._rng.sample(self.pool, batch_size)
```

替换为下面这段，后面构造 `TransitionBatch` 的代码保持不变：

```python
def tensor_key(t):
    return (
        tuple(t.shape),
        t.dtype,
        tuple(t.detach().cpu().reshape(-1).tolist()),
    )

unique = {}
for t in self.pool:
    key = (
        tensor_key(t.state),
        t.action,
        float(t.reward),
        tensor_key(t.next_state),
        t.terminated,
        t.truncated,
    )
    unique[key] = t

if len(unique) < batch_size:
    raise ValueError(
        f"不同 transition 只有 {len(unique)} 个，"
        f"不足以无放回采样 {batch_size} 个"
    )

batch = self._rng.sample(list(unique.values()), batch_size)
```

`unique[key] = t` 会让内容相同的 transition 只保留一个代表。因此，不管它原来出现一次还是一百次，抽样机会都一样。原来的 `self.pool` 不会被修改。

**但这个版本只适合小规模验证：你目前 buffer 容量是 500,000，每次更新都遍历整个 buffer、转换 Tensor，会非常慢。**

正式训练更合适的做法是：在 `push()` 时维护不同 transition 的索引和副本数量，淘汰时同步更新，`sample()` 直接抽索引。这就是之前实验中的[实现](./analysis/epsilon_mab_20260908/replay_weighting.py)。代码会多一点，但避免每个训练步重新扫描 50 万条记录。

> 那我觉得，我下面这个想法应该和这样子差不多吧：
>
> 就是我在 push 的时候，我先检查一下里面有没有，如果有的话，我就不 push 进去了

对，**就“让不同 transition 获得相同抽样机会”而言，你这个方法可以，而且更直接**：`push()` 时去重，`sample()` 继续均匀抽样。

但它和刚才的方案有一个区别：**buffer 满了之后，淘汰行为不同。**

| 方案 | 容量 500,000 表示什么 |
|---|---|
| 保留重复记录，抽样时等权 | 最近 500,000 条经验中的不同 transition |
| `push()` 时拒绝重复 | 最近接纳的 500,000 个不同 transition |

你的方案中，重复经验不会挤掉旧经验，因此旧 transition 可能保留更久。对于当前确定性迷宫，这是一个值得测试的简化方案，但不能直接套用之前实验的成功率。

实现时注意两点：

- **检查完整 transition 的内容**，包括 `state、action、reward、next_state、terminated、truncated`，不能只检查 state。
- 用一个额外的 `set` 保存不可变的内容 key，快速判断重复。不要直接写 `transition in self.pool`，Tensor 的相等比较不适合这样使用；buffer 淘汰旧元素时，也要删除对应 key。

另外，拒绝重复后，实际 buffer 长度可能小于 `batch_size`，开始采样前要检查长度。

> 你去验证一下，看一下我这个想法的性能怎么样？

<details><summary>22 previous messages</summary>

> 我会用相同的 seed 42 和训练配置验证“push 时拒绝重复 transition”，与之前两种 replay 方式比较成功率、训练耗时和实际 buffer 大小。
>
> <details><summary>Explored 3 files, a search, 3 lists, ran 6 commands</summary>
>
> <details><summary>Explored 3 files, a search, 3 lists</summary>
>
> - Searched for `Single.variable|single-variable|replay|epsilon` in `MEMORY.md`
> - Listed files in `epsilon_mab_20260908`
> - Read `./analysis/epsilon_mab_20260908/replay_experiment.py`
> - Listed files in `analysis`
> - Read `./analysis/epsilon_mab_20260908/replay_weighting.py`
> - Read `./analysis/epsilon_mab_20260908/all_mab_unique_experiment.py`
> - Listed files
>
> </details>
>
> - Ran `cat > analysis/epsilon_mab_20260908/push_unique_replay.py <<'PY'
> """Reject duplicates at insertion; duplicates do not refresh FIFO age."""
> import collections
> from replay_weighting import IndexedReplay, transition_key
> from IncompleteBaseline.algorithms.common.replay_buffer import ReplayBuffer, Transition
>
> class PushUniqueReplay(IndexedReplay):
>  def __init__(self, buffer_size, seed, input_dim):
>   ReplayBuffer.__init__(self,buffer_size,seed,input_dim)
>   self.pool=[];self.unique=True;self._active=[];self._reps={};self._cursor=0
>   self._sample_counts=collections.Counter();self._sample_total=0;self._sample_positive=0;self._sample_selfloop=0
>   self.attempts=0;self.rejected=0;self.evicted=0
>  def push(self,t):
>   assert isinstance(t,Transition)
>   assert t.state.shape==t.next_state.shape==(1,self.input_dim)
>   self.attempts+=1;k=transition_key(t)
>   if k in self._reps:
>    self.rejected+=1;return
>   if len(self.pool)<self.buffer_size:
>    self.pool.append(t);self._active.append(k)
>   else:
>    old=self._active[self._cursor];del self._reps[old]
>    self.pool[self._cursor]=t;self._active[self._cursor]=k
>    self._cursor=(self._cursor+1)%self.buffer_size;self.evicted+=1
>   self._reps[k]=t
>  def diagnostics(self):
>   total=self._sample_total
>   r={'buffer_entries':len(self.pool),'buffer_unique_transitions':len(self._active),'push_attempts':self.attempts,'rejected_duplicates':self.rejected,'evictions':self.evicted,'sampled_entries':total,'sampled_unique_transitions':len(self._sample_counts),'sampled_top10_fraction':sum(x for _,x in self._sample_counts.most_common(10))/total if total else 0,'sampled_positive_fraction':self._sample_positive/total if total else 0,'sampled_selfloop_fraction':self._sample_selfloop/total if total else 0}
>   self._sample_counts.clear();self._sample_total=self._sample_positive=self._sample_selfloop=0
>   return r
> PY
> /Users/jay/miniconda3/envs/rl/bin/python - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> s=(p/'all_mab_unique_experiment.py').read_text().replace('from replay_weighting import IndexedReplay','from push_unique_replay import PushUniqueReplay')
> s=s.replace("default=['all_mab_unique_replay']","default=['eps_push_unique','all_mab_push_unique']")
> s=s.replace("manifest={'seed':42", "base.replaybuffer=PushUniqueReplay(500000,SEED,45)\nmanifest={'duplicate_policy':'reject_without_refresh_from_start_of_warmup','seed':42")
> s=s.replace("'all_mab_unique_manifest.json'","'push_unique_manifest.json'")
> s=s.replace("a.replaybuffer.pool=collections.deque(base.replaybuffer.pool,maxlen=500000)","a.replaybuffer.pool=list(base.replaybuffer.pool)")
> s=s.replace(" a.replaybuffer=IndexedReplay(a.replaybuffer,unique=(mode in ['eps_unique_replay','all_mab_unique_replay']))", " a.replaybuffer._active=list(base.replaybuffer._active);a.replaybuffer._reps=dict(base.replaybuffer._reps);a.replaybuffer._sample_counts=collections.Counter()")
> s=s.replace("['all_mab','all_mab_unique_replay']","['all_mab_push_unique']")
> s=s.replace("# Freeze the warmup checkpoint", "(OUT/'push_unique_warmup.json').write_text(json.dumps(base.replaybuffer.diagnostics(),indent=2))\n# Freeze the warmup checkpoint")
> (p/'push_unique_experiment.py').write_text(s)
> PY`
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python - <<'PY'
> import sys,random,collections
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908');sys.path[:0]=[str(p),str(p/'frozen_project')]
> import torch
> from push_unique_replay import PushUniqueReplay
> from replay_weighting import transition_key
> from IncompleteBaseline.algorithms.common.replay_buffer import Transition
> b=PushUniqueReplay(7,42,1);ref=collections.OrderedDict();r=random.Random(1)
> for i in range(500):
>  v=r.randrange(12);t=Transition(torch.tensor([[float(v)]]),0,0.,torch.tensor([[float(v+1)]]),False,bool(i%2));k=transition_key(t)
>  if k not in ref:
>   if len(ref)==7:ref.popitem(last=False)
>   ref[k]=t
>  b.push(t)
>  assert set(ref)==set(b._reps)==set(b._active)
>  if len(ref)>=4:
>   batch=b.sample(4);assert len(set(zip(batch.states.flatten().tolist(),batch.truncated.flatten().tolist())))==4
> print('PASS: 500 insertions, rejection without refresh, FIFO eviction, sample uniqueness, truncated distinction')
> PY`
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -u analysis/epsilon_mab_20260908/push_unique_experiment.py > analysis/epsilon_mab_20260908/push_unique.log 2>&1`
>
> - Ran `cat analysis/epsilon_mab_20260908/eps_unique_replay_done.json analysis/epsilon_mab_20260908/all_mab_unique_replay_done.json`
>
> - Ran `tail -4 analysis/epsilon_mab_20260908/push_unique.log`
>
> - Ran `git -C project/IncompleteBaseline status --short`
>
> </details>
>
> 已按你的想法实现实验版：重复 transition 直接拒绝，也不刷新它在 FIFO 中的位置。去重从 warmup 开始生效。
>
> 重复检查、容量淘汰和 batch 内无重复的测试都已通过。接下来两组各训练 120,000 步：ε-greedy + MAB，以及全程 MAB；正式项目代码保持不变。
>
> <details><summary>Explored 3 files, ran 4 commands</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./analysis/epsilon_mab_20260908/replay_followup_plot.py`
>
> </details>
>
> - Ran `tail -3 analysis/epsilon_mab_20260908/push_unique.log`
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./analysis/epsilon_mab_20260908/replay_trajectory_audit.py`
> - Read `./analysis/epsilon_mab_20260908/replay_policy_audit.py`
>
> </details>
>
> - Ran `cat > analysis/epsilon_mab_20260908/push_unique_report.py <<'PY'
> from pathlib import Path
> import json,os,hashlib
> import numpy as np
> os.environ.setdefault('MPLCONFIGDIR','/tmp/mab-mpl-cache')
> import matplotlib
> matplotlib.use('Agg')
> import matplotlib.pyplot as plt
> P=Path(__file__).resolve().parent
> pairs=[['eps_live','eps_unique_replay','eps_push_unique'],['all_mab','all_mab_unique_replay','all_mab_push_unique']]
> labels=['Original replay','Equal values in recent FIFO','Reject duplicates on push']
> colors=['#888888','#007f73','#d56b25']
> plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
> fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
> summary={}
> for ax,modes,title in zip(axs,pairs,['epsilon-greedy + MAB','Full MAB']):
>  for m,label,c in zip(modes,labels,colors):
>   e=json.loads((P/f'{m}_eval.json').read_text());w=json.loads((P/f'{m}_windows.json').read_text());done=json.loads((P/f'{m}_done.json').read_text())
>   loss=np.load(P/f'{m}_loss.npy');assert len(loss)==120000 and np.isfinite(loss).all()
>   assert e[-1]['step']==120000
>   ax.plot([r['step']/1000 for r in e],[r['fixed']['heldout']['success'] for r in e],label=label,color=c,lw=2)
>   summary[m]={'fixed':e[-1]['fixed'],'official':e[-1]['official'],'training_seconds_including_eval_io':done['elapsed'],'training_goals':done['total_solved'],'last20k_unique_states_per5k':float(np.mean([r['unique_states'] for r in w[-4:]])),'last20k_top10_fraction':float(np.mean([r['top10_fraction'] for r in w[-4:]]))}
>   if 'replay_sampling' in w[-1]:summary[m]['final_replay']=w[-1]['replay_sampling']
>  ax.set(title=title,xlabel='Training steps (thousands)',ylabel='Held-out success rate (fixed 100 mazes)',ylim=(0,1))
> fig.suptitle('Rejecting duplicate transitions at insertion — seed 42')
> fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
> fig.savefig(P/'push_unique_learning.png',dpi=170);fig.savefig(P/'push_unique_learning.pdf');plt.close(fig)
> old=json.loads((P/'all_mab_windows.json').read_text());new=json.loads((P/'all_mab_push_unique_windows.json').read_text())
> ignore={'loss_mean','replay_sampling'}
> mismatches=[(i+1,k) for i,(a,b) in enumerate(zip(old,new)) for k in a if k not in ignore and a[k]!=b.get(k)]
> assert len(old)==len(new)==24 and not mismatches
> trajectories=json.loads((P/'push_unique_trajectory_audit.json').read_text())
> for m in pairs[0][2:]+pairs[1][2:]:
>  for split in ['train','heldout']:
>   assert trajectories[m][split]['counts'].get('success',0)==round(100*summary[m]['fixed'][split]['success'])
> warm=json.loads((P/'push_unique_warmup.json').read_text())
> audit={'steps_per_mode':120000,'finite_losses_all_six_modes':True,'new_models_reloaded_fixed_eval_matches':True,'all_mab_behavior_windows_compared':24,'all_mab_behavior_mismatches':mismatches,'warmup':warm,'source_hashes':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['push_unique_replay.py','push_unique_experiment.py']}}
> (P/'push_unique_audit.json').write_text(json.dumps(audit,indent=2));(P/'push_unique_summary.json').write_text(json.dumps(summary,indent=2))
> def pct(m):return f"{100*summary[m]['fixed']['heldout']['success']:.0f}%"
> e=summary['eps_push_unique'];a=summary['all_mab_push_unique']
> text=f'''**push 时拒绝重复 transition：seed 42 对照实验**
>
> 保持之前的 frozen_project 源码和参数，新增两组：从 warmup 第一条经验开始拒绝重复完整 transition。重复不刷新 FIFO 年龄。容量满时淘汰最早接纳、仍在 buffer 内的一条不同经验。完整 key 包含 state/next_state 的数值、形状和 dtype，以及 action、reward、terminated、truncated。
>
> | Replay 方式 | epsilon-greedy + MAB | 全程 MAB |
> |---|---:|---:|
> | 原记录均匀抽样 | {pct('eps_live')} | {pct('all_mab')} |
> | 最近 FIFO 内不同 transition 等权 | {pct('eps_unique_replay')} | {pct('all_mab_unique_replay')} |
> | push 时拒绝重复 | {pct('eps_push_unique')} | {pct('all_mab_push_unique')} |
>
> 以上为相同的 100 个固定未见迷宫、每局 25 步、纯贪心成功率。seed=42，warmup=819199 个环境步，随后训练 120000 步，batch=64，buffer 上限=500000，train episode limit=2500、warmup limit=5000。epsilon 调度保持原样，在训练 120000 步时为 0.443359375，探索用 MAB 并更新 MAB。两个新增组来自同一个 warmup 检查点。
>
> ![学习曲线]({P}/push_unique_learning.png)
>
> **buffer 实际规模与保留历史**
>
> warmup 819199 次 push 中拒绝 {warm['rejected_duplicates']:,} 次，实际留下 {warm['buffer_entries']:,} 条不同 transition，淘汰 {warm['evictions']:,} 条。训练结束时 epsilon 组有 {e['final_replay']['buffer_entries']:,} 条，累计拒绝 {e['final_replay']['rejected_duplicates']:,} 次；全程 MAB 组有 {a['final_replay']['buffer_entries']:,} 条，累计拒绝 {a['final_replay']['rejected_duplicates']:,} 次。两组累计淘汰分别为 {e['final_replay']['evictions']}、{a['final_replay']['evictions']}。
>
> 这与上一种等权 sampler 有两个差异：可抽样集合由不同经验容量决定，而非最近 500000 个环境步；不同集合及其索引顺序也改变了相同 RNG 下的具体 batch。因此它们即使都按不同值等权，也不应预期逐步 loss 相同。结果差异不能进一步单独归因于保留历史这一项。
>
> **运行时间与实现**
>
> 新增 epsilon 组训练阶段实测 {e['training_seconds_including_eval_io']:.1f} 秒，全程 MAB 组 {a['training_seconds_including_eval_io']:.1f} 秒，包括定期评估与保存文件，不包括 warmup。历史等权 sampler 两组记录分别为 {summary['eps_unique_replay']['training_seconds_including_eval_io']:.1f}、{summary['all_mab_unique_replay']['training_seconds_including_eval_io']:.1f} 秒。运行时机器负载可能不同，这些数字只说明本次实际耗时，不是严格的速度排名。
>
> 实验使用内容 key 字典检查重复，列表环形 FIFO 保持 O(1) 淘汰，并复用之前 sampler 的 batch 构造和采样诊断。没有在每次 push 或 sample 中扫描整个 buffer；sample 从不同 transition 索引直接均匀无放回抽样。
>
> **核验与适用范围**
>
> 500 次小容量插入与 OrderedDict 参考实现核对，覆盖拒绝重复、不刷新年龄、FIFO 淘汰、终止字段区分及无重复 batch。训练 loss 均有限且各有 120000 点；重新加载模型核对固定迷宫成功数一致。全程 MAB 新旧两组 24 个窗口的所有已记录行为统计完全一致，验证 replay 变化没有改变该组的数据采集过程。
>
> 这是一个 seed、120000 训练步、确定性 5×5 迷宫的实验，不能据此判定两个高成功率方案统计等价或确定长期排名。项目训练代码未修改，新增实验和报告均在 analysis 目录。
>
> [数值汇总]({P}/push_unique_summary.json)、[验证记录]({P}/push_unique_audit.json)、[实验 replay 实现]({P}/push_unique_replay.py)、[训练脚本]({P}/push_unique_experiment.py)。
> '''
> (P/'push_unique_report.md').write_text(text)
> print(json.dumps(summary,indent=2));print('AUDIT PASSED')
> PY
> /Users/jay/miniconda3/envs/rl/bin/python - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> s=(p/'replay_trajectory_audit.py').read_text().replace("['all_mab','eps_live','eps_unique_replay','all_mab_unique_replay']","['eps_push_unique','all_mab_push_unique']").replace("'replay_trajectory_audit.json'","'push_unique_trajectory_audit.json'")
> (p/'push_unique_trajectory_audit.py').write_text(s)
> PY`
>
> - Ran `tail -5 analysis/epsilon_mab_20260908/push_unique.log`
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python - <<'PY'
> import json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> print('warmup',json.loads((p/'push_unique_warmup.json').read_text()))
> for m in ['eps_live','eps_unique_replay','eps_push_unique']:
>  e=json.loads((p/f'{m}_eval.json').read_text());print(m,[(x['step'],x['fixed']['heldout']['success']) for x in e if x['step']<=30000])
> PY`
>
> </details>
>
> 前 20,000 个训练步已完成：你的方案在固定未见迷宫上的成功率为 13%，之前“抽样时等权”为 12%，原 replay 为 0%。目前两种去重方案表现接近，继续跑满预算再比较。
>
> warmup 的 819,199 次 push 中拒绝了 477,070 次重复，留下 342,129 条不同经验，还没达到 500,000 的容量上限。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `tail -4 analysis/epsilon_mab_20260908/push_unique.log`
> - Ran `cat > analysis/epsilon_mab_20260908/check_push_unique.py <<'PY'
> import sys,random,collections,json
> from pathlib import Path
> P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'frozen_project'))
> import torch
> from push_unique_replay import PushUniqueReplay
> from replay_weighting import transition_key
> from IncompleteBaseline.algorithms.common.replay_buffer import Transition
> b=PushUniqueReplay(7,42,1);ref=collections.OrderedDict();r=random.Random(1)
> for i in range(500):
>  v=r.randrange(12);t=Transition(torch.tensor([[float(v)]]),0,0.,torch.tensor([[float(v+1)]]),False,bool(i%2));k=transition_key(t)
>  if k not in ref:
>   if len(ref)==7:ref.popitem(last=False)
>   ref[k]=t
>  b.push(t)
>  assert set(ref)==set(b._reps)==set(b._active)
>  order=b._active[b._cursor:]+b._active[:b._cursor]
>  assert order==list(ref)
>  if len(ref)>=4:
>   batch=b.sample(4);assert len(set(zip(batch.states.flatten().tolist(),batch.truncated.flatten().tolist())))==4
> result={'insertions_checked':500,'contents_and_fifo_order_match_reference':True,'batches_contain_unique_values':True,'attempts':b.attempts,'rejected':b.rejected,'evicted':b.evicted}
> (P/'push_unique_unit_check.json').write_text(json.dumps(result,indent=2));print(result)
> PY
> /Users/jay/miniconda3/envs/rl/bin/python analysis/epsilon_mab_20260908/check_push_unique.py`
>
> </details>
>
> 训练到 50,000 步，你的方案成功率已达到 43%。目前没有出现 buffer 样本不足或数值异常。小容量测试也确认：重复插入不会刷新 FIFO 年龄，淘汰顺序与参考实现一致。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `tail -5 analysis/epsilon_mab_20260908/push_unique.log`
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python - <<'PY'
> import json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> for m in ['eps_unique_replay','eps_push_unique','all_mab_unique_replay']:
>  w=json.loads((p/f'{m}_windows.json').read_text())
>  print(m,'first recorded replay entries/unique',w[0]['replay_sampling']['buffer_entries'],w[0]['replay_sampling']['buffer_unique_transitions'],'latest',w[-1]['step'],w[-1]['replay_sampling']['buffer_entries'])
> PY`
>
> </details>
>
> 到 80,000 步，成功率升到 69%。你的方案确实保留了更多不同经验：训练早期约 34.3 万条，而之前“最近 50 万条记录内等权抽样”的可选不同经验约 19.4 万条。两种方案的训练数据集合已经不同，最终结果不必完全一致。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `tail -5 analysis/epsilon_mab_20260908/push_unique.log`
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python -m py_compile analysis/epsilon_mab_20260908/push_unique_report.py analysis/epsilon_mab_20260908/push_unique_trajectory_audit.py`
>
> </details>
>
> ε-greedy 组已接近完成，110,000 步时成功率为 79%。全程 MAB 对照随后继续运行，用来检查这一简化方案在另一种采集策略下的效果。
>
> <details><summary>Explored a file, ran 2 commands</summary>
>
> - Ran `tail -7 analysis/epsilon_mab_20260908/push_unique.log`
>
> <details><summary>Explored a file</summary>
>
> - Read `./analysis/epsilon_mab_20260908/eps_push_unique_done.json`
>
> </details>
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python - <<'PY'
> import json
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> e=json.loads((p/'eps_push_unique_eval.json').read_text())[-1];w=json.loads((p/'eps_push_unique_windows.json').read_text())[-1]
> print(e);print(w['replay_sampling'])
> old=json.loads((p/'all_mab_windows.json').read_text());new=json.loads((p/'all_mab_push_unique_windows.json').read_text());ignore={'loss_mean','replay_sampling'}
> print('fullMAB windows checked',len(new),'mismatches',[(i,k) for i,(a,b) in enumerate(zip(old,new)) for k in a if k not in ignore and a[k]!=b.get(k)])
> PY`
>
> </details>
>
> ε-greedy 组跑满了 120,000 步：固定未见迷宫成功率为 **77%**，之前两种方式分别为 **1% 和 71%**。最终 buffer 留下 378,563 条不同经验，尚未发生容量淘汰。
>
> 这说明你的想法在本次实验中有效；77% 和 71% 的差距还不能说明它稳定更优。全程 MAB 对照仍在运行，已完成窗口的采集统计与原实验完全一致。
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `tail -4 analysis/epsilon_mab_20260908/push_unique.log`
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908/push_unique_report.py')
> s=p.read_text();needle='以上为相同的 100 个固定未见迷宫'
> s=s.replace(needle, "原协议随机评估的未见迷宫成功率：epsilon 组三种方式分别为 {100*summary['eps_live']['official']['heldout']['success']:.0f}%、{100*summary['eps_unique_replay']['official']['heldout']['success']:.0f}%、{100*summary['eps_push_unique']['official']['heldout']['success']:.0f}%；全程 MAB 三种方式分别为 {100*summary['all_mab']['official']['heldout']['success']:.0f}%、{100*summary['all_mab_unique_replay']['official']['heldout']['success']:.0f}%、{100*summary['all_mab_push_unique']['official']['heldout']['success']:.0f}%。因此不根据某一个评估子集的几个百分点差异判定两种等权方案谁稳定更优。\n\n"+needle)
> p.write_text(s)
> PY`
>
> </details>
>
> 全程 MAB 对照训练到 60,000 步，成功率为 54%，也在持续改善。报告会同时列出固定评估和随机评估结果：目前两种去重方案在不同评估子集上的排名会变化，不能只凭一个数字选赢家。
>
> <details><summary>Ran a command</summary>
>
> - Ran `tail -4 analysis/epsilon_mab_20260908/push_unique.log`
>
> </details>
>
> 全程 MAB 组的评估有波动：60,000 步是 54%，80,000 步回落到 47%。我会按预定的 120,000 步结果比较，不挑中途最高点。完成后还会重新加载保存的模型，核对成功率记录。
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `tail -4 analysis/epsilon_mab_20260908/push_unique.log`
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python analysis/epsilon_mab_20260908/push_unique_trajectory_audit.py`
> - Ran `tail -4 analysis/epsilon_mab_20260908/push_unique.log`
>
> </details>
>
> 两组都已完成：你的方案最终在 ε-greedy 下达到 **77%**，全程 MAB 下达到 **76%**；之前抽样时等权的结果分别为 71% 和 72%。两组各训练约 4.3 分钟。
>
> 正在做最后的模型重载核验和图表检查。从当前结果看，push 时去重是可行的简化方向。
>
> <details><summary>Ran a command</summary>
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python analysis/epsilon_mab_20260908/push_unique_report.py > analysis/epsilon_mab_20260908/push_unique_report_stdout.json`
>
> </details>
>
> Viewed an image
>
> <details><summary>Explored a file, ran 2 commands</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `./analysis/epsilon_mab_20260908/push_unique_report.md`
>
> </details>
>
> - Ran `git -C project/IncompleteBaseline status --short`
>
> - Ran `/Users/jay/miniconda3/envs/rl/bin/python - <<'PY'
> from pathlib import Path
> p=Path('analysis/epsilon_mab_20260908')
> for f in ['push_unique_report.py','push_unique_report.md']:
>  q=p/f;s=q.read_text().replace('以上为相同的 100 个固定未见迷宫、每局 25 步、纯贪心成功率。','表格与曲线使用相同的 100 个固定未见迷宫、每局 25 步、纯贪心成功率；上面的随机评估结果单独列出。');q.write_text(s)
> PY`
>
> Tool call
>
> Tool: open_in_codex
> Status: completed
>
> </details>
</details>

验证完成。**你的想法有效：push 时拒绝重复，明显改善了本次训练，效果与之前的等权采样接近。**

保持 seed 42、相同 warmup 和参数，随后训练 120,000 步，在相同的 100 个未见迷宫上评估：

| Replay 方式 | ε-greedy + MAB | 全程 MAB |
|---|---:|---:|
| 原来的均匀抽记录 | 1% | 26% |
| 抽样时对不同 transition 等权 | 71% | 72% |
| **你的方案：push 时拒绝重复** | **77%** | **76%** |

![学习曲线](./analysis/epsilon_mab_20260908/push_unique_learning.png)

实际运行情况：

- ε-greedy 组累计尝试插入 **939,199 次**，拒绝 **560,636 次**重复，留下 **378,563 条**不同经验。
- 没达到 500,000 的容量上限，**本次训练尚未触发容量淘汰**。
- 两组训练分别耗时 **262 秒、257 秒**，包含评估和保存，不含 warmup。
- loss 均有限；重新加载模型后，评估成功数与记录一致。

**可以把它作为当前迷宫任务的简化实现。** 不过，这只是一个 seed 的结果，还不能说它稳定优于之前的等权 sampler；换用随机评估子集时，两者排名会变化。

正式项目代码未修改。[完整报告](./analysis/epsilon_mab_20260908/push_unique_report.md) · [实验实现](./analysis/epsilon_mab_20260908/push_unique_replay.py)
