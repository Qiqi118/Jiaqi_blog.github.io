# Spine：An Efficient DRL-based Congestion Control with Ultra-low

## Abstract

        先前基于深度强化学习（Deep Reinforcement Learning，DRL）的拥塞控制（CC）算法直接调整流发送速率以响应动态带宽变化，会产生较大的开销，进而会消耗大量的CPU资源。在本文中，我们提出了一种分层拥塞控制算法Spine，它充分利用了深度强化学习的性能增益，但开销极低。在其核心，Spine将拥塞控制任务分解为不同时间尺度的两个子任务，并使用不同的组件来处理它们：i）一个轻量级CC执行器，它响应动态带宽变化执行细粒度控制；ii）一个RL代理，它在粗粒度级别工作，为CC执行器生成控制子策略。这样的两级控制架构可以提供具有低模型推理开销的基于细粒度DRL的控制。

## 1、Introduction

        基于DRL的CC方案能够通过单一控制策略来适应不同的网络条件。因此，网络工程师可以摆脱手动调整CC超参数以适应未知网络条件的操作挑战。以前基于DRL的CC方案由于发送速率调整的复杂模型推断而产生了较高的计算开销。现有解决方案通过降低推断频率以延长响应间隔来处理开销问题。在间隔期间，CC不受DRL控制，要么不进行速率调整，要么依赖于经典方案，例如Orca中的Cubic。因此，它们无法充分利用DRL模型提供的性能优势，并且由于粗粒度控制，容易受到网络拥塞的影响。

        我们思考一个问题：基于DRL的CC能否在保持低计算开销的同时为每个ACK提供细粒度控制？

        Spine采用了一种分层控制架构，该架构由一个轻量级CC executor（对每个ACK和丢失事件做出反应）和一个基于DRL的policy generator（周期性地为CC执行器生成控制子策略，以适应网络条件的变化，例如带宽容量变化或流量到达和离开）组成。Spine中的子策略是由policy generator输出的一组参数定义AIMD方法。与网络事件反应（例如，ACK和分组丢失）相比，子策略的更新频率较低，进而降低DRL模型推断的频率。因此，Spine可以以较低的计算开销执行基于细粒度DRL的CC。

        为了保证及时的网络适应，同时保持较低的模型推理开销，Spine通过进一步引入观察者模块（与策略生成器相比，该模块是一个较小的模型），利用了灵活的子策略更新策略。观察者判断当前子策略是否仍然有效，并在必要时触发策略生成器更新它。因此，这种灵活的更新策略显著降低了策略生成器的执行频率以及子策略更新操作所涉及的潜在跨空间（内核和用户空间）通信（尤其是在稳定的网络条件下）。

# 2、Motivation

### 2.1 DRL-based Congestion Control

        与专注于预测和分类任务的监督学习算法不同，强化学习通过在与环境的交互过程中长期最大化累积回报来处理顺序决策过程。借助深度神经网络，深度强化学习在许多游戏和现实世界决策任务中发挥着核心作用。

### 2.2 Overhead vs. Performance

        尽管有前景，但基于DRL的模型比经典CC方案中的简单ACK响应函数花费的时间和消耗的计算资源要多几个数量级。采用这种DRL的方法可能没有足够的CPU资源来处理，进而导致吞吐量下降，此外，考虑到存在较高的模型推断延迟，这种基于DRL的CC方案很难采用小的控制间隔。然而，采用具有粗粒度控制的基于DRL的CC方案无法充分释放深度强化学习的潜力。

        改善开销和性能之间的权衡的一种方法是将经典方案重新纳入细粒度控制。Orca通过构建两级控制框架实现了这一点，通过RL模型和Cubic共同控制𝑐𝑤𝑛𝑑。

        以前所有基于DRL的方案都集中于直接调整当前发送速率。因此，RL代理需要执行两个子任务：i）以较高频率更新发送速率，以及时响应动态带宽变化；ii）当链路改变或流到达和离开时，使其当前事件动作映射适应网络条件的改变。

        基于DRL的方案擅长于子任务（ii），因为它们能够推广到各种网络条件，但开销问题使它们无法快速响应动态带宽变化，从而导致次优性能。另一方面，经典方案由于其简单的硬连线信号动作映射而擅长于子任务（i），但难以自动适应其假设不再成立的网络条件。因此，我们认为，为了以低开销实现高模型性能，我们需要对CC任务进行解耦，并采用分层架构来在不同的时间尺度上处理这两个子任务。

### 2.3 Key Design Decisions

        我们的关键设计决策是将耗时的DRL处理与快速发送速率调整分开。Spine中的DRL用于学习生成一个子策略而不是直接学习更新发送速率。子策略可以被视为从分组级事件到发送速率调整的参数化映射，与DRL模型相比，该映射能够在细粒度级别即时响应网络信号。

        DRL代理定期观察当前网络状况并生成子策略。DRL在粗粒度级别不断根据网络装填的变化更新子策略。这种方式给解决性能和开销之间的矛盾带来了几个好处。

    1）DRL通过生成子策略而不是极易发生变化的cwnd和发送速率，大大降低了DRL模型的工作频率，因此开销比其他基于DRL的CCA降低了很多。

    2）通过学习生成合适的子策略以在细粒度级别控制每一个响应，Spine在各种网络条件下实现了一致的高性能，即使在链路容量急剧变化的动态网络条件下也是如此。

    3）分层策略架构支持更灵活的更新策略。在检查网络状况后，DRL可以判断当前子策略是否仍然正常工作，从而减少更新子策略的次数。进一步降低稳定网络条件下的开销。

## 3 Design

### 3.1 Overview

        如下图所示，Spine由三个模块组成，策略生成器（policy generator）、观察者（watcher）和一个CC执行器（CC executor）。

<img src="file:///F:/qiqiblog.github.io/images/2022.11.21-Spine-An-Efficient-DRL_based-Congestion-Control-with-Ultra_low/high-level%20architecture%20of%20Spine%20.png" title="" alt="avatar" data-align="center">

        策略生成器和观察者一起形成RL Agent，该Agent跟踪流量模式并更新当前子策略。CC执行器在内核中实现，以执行RL Agent生成的控制子策略，调整响应于ACK和分组丢失的发送速率。观察者根据搜集的数据包统计信息判断子策略在当前的网络条件下是否仍然正常工作。如果正常工作，则RL Agent不采取任何动作。否则触发策略生成模块并向其提交当前网络条件信息。策略生成器被触发之后，根据收到的网络条件信息生成新的子策略并更新CC执行器，此外还要更新观察者以便继续监督新的子策略。

        如下图所示，策略生成器、观察者和CC执行者在不同的时间尺度上运行。对于每个监视间隔（monitor interval，MI），观察者将数据包统计信息作为RL代理的当前状态输入进行观察，并每隔一段时间触发一次策略生成器。最后，策略生成器以灵活的信号驱动方式工作：它只在触发时更新观察者和执行者。与策略生成器相比，观察者通常更小，因此与以前基于DRL的方案相比，具有更小的常规计算成本。

<img title="" src="file:///F:/qiqiblog.github.io/images/2022.11.21-Spine-An-Efficient-DRL_based-Congestion-Control-with-Ultra_low/time%20diagram%20of%20Spine’s%20hierarchical%20control.png" alt="avatar" data-align="center" width="405">

### 3.2 RL Agent

        RL Agent的工作方式：在每个监视间隔（MI），通过收集分组统计信息来感知网络状态，以此作为Agent的当前状态（state）。然后将该状态输入到模型中，该模型通过设置一个trigger标志来决定是否需要更新子策略。如果trigger是真，模型会通过生成参数设置$a_t$并将其发送到内核中更新CC执行器。

**State** 输入到RL Agent的state包括上个MI以来收集的流的包的状态。我们考虑的状态信息，如下表所示。除此之外，我们还将当前部署的子策略包括到状态中，因此Spine可以评估当前子策略的性能，以进行更智能的更新决策。

<img title="" src="file:///F:/qiqiblog.github.io/images/2022.11.21-Spine-An-Efficient-DRL_based-Congestion-Control-with-Ultra_low/packet%20statistics%20used%20as%20Spine’s%20RL%20agent%20input.png" alt="avatar" width="482" data-align="center">

        一些先前的DRL CC方案堆叠固定长度的历史特征，使得Agent可以通过从历史分组统计中提取信息来更精确地推断当前网络状况。在Spine中，我们采用递归神经网络（RNN）作为分层策略结构的构建块。RNN能够从长期历史特征中捕获模式和依赖性，而无需堆叠特征，这对于策略生成器和观察者在稀疏触发事件中记忆其状态历史非常重要。因此，我们可以直接将当前状态特征馈送到模型中，而无需状态堆叠。

**Reward** 

$$
R=(\frac{thr-\zeta\times loss}{lat'})/(\frac{thr_{max}}{lat_{min}})-\alpha_{psp}\times trigger
$$

其中，

$$
lat'=
\begin{cases}
lat_{min},& (lat_{min}\leq lat\leq \beta\times lat_{min})\\
lat,& otherwise.
\end{cases}

$$

        reward的后半部分触发策略生成器更新新子策略的惩罚，因为这将导致策略生成器和子策略更新的跨空间通信的进一步推断开销。我们将处罚称为pit stop penalty，因为更改子政策会产生额外费用，如果没有必要，我们希望避免，这类似于赛车运动中更换轮胎的情况。如果没有pit stop penalty，RL代理即使在更新产生的作用很小的情况下，也无法减少在MI更新子策略的次数。指示器𝑡𝑟𝑖𝑔𝑔𝑒𝑟 当观察者触发策略生成器时，等于1，否则为0。系数𝛼𝑝𝑠𝑝 定义了惩罚的意义。

### 3.3 CC Executor

        Spine中的CC执行器被实现为可插拔拥塞控制模块之一，类似于Cubic和BBR，但执行由策略生成器生成的子策略。为了实现Spine的设计目标，我们希望我们的参数化子策略结构具有以下特征：

    1）**简单**。子策略应该足够简单，以便CC执行器能够在内核中以非常低的计算开销执行。

    2）**细粒度控制**。子策略应该细粒度的控制发送速率或𝑐𝑤𝑛𝑑 以快速响应动态带宽变化。

    3）**灵活**。灵活的子策略能够近似从信号到发送速率的各种控制映射，能够学习最优策略。

        我们基于AIMD的思想设计了一个简单而有效的子策略。它采用了拥塞控制中最常用的三个指标的组合：接收ACK、数据包延迟和丢失。在执行子策略的同时，CC执行器还在开始时执行一个额外的慢启动，发送方每RTT将其发送速率乘以1.1，直到发生数据包丢失，这与Orca类似。对于子策略执行部分，当接收到ACK包时，CC执行器使用以下公式更新𝑐𝑤𝑛𝑑。

$$
\Delta cwnd=
\begin{cases}
-\alpha_{lat} & \frac{RTT}{RTT_{min}}\geq\alpha_{tol}+1 \\
\alpha_{thr} & otherwise,
\end{cases}

$$

        其中，$0\leq\alpha_{thr}$, $\alpha_{lat}\leq0.5$, $0\leq\alpha_{tol}\leq2$是超参数（hyperparameter）。如果$\frac{RTT}{RTT_{min}}$小于等于$\alpha_{tol}+1$则认为没有发生拥塞，通过$\alpha_{thr}$增加cwnd，否则通过$\alpha_{lat}$减小cwnd。$\alpha_{thr}$增加cwnd控制攻击性，$\alpha_{lat}$控制排队延迟的敏感性，$\alpha_{tol}$决定排队容忍度的目标延迟点。

        当发生数据包丢失时，CC执行器执行𝑐𝑤𝑛𝑑 乘以$\alpha_{𝑙𝑜𝑠𝑠}$ , 类似于Cubic：

$$
cwnd_{new}=\alpha_{loss}\times cwnd,0\leq\alpha_{loss}\leq1
$$

        其中$\alpha_{𝑙𝑜𝑠𝑠}$表示对丢包的敏感性。在更新cwnd之后，CC执行器根据如下公式计算新的pacing rate。

$$
P_{rate}=\frac{cwnd}{RTT}
$$

        使用上述子策略结构，通过参数设置（$\alpha_{thr},\alpha_{lat},\alpha_{tol},\alpha{loss}$）确定子策略的行为。我们将上述参数集定义为策略生成器的操作输出𝑎𝑡。通过控制参数设置，Spine定制子策略对不同信号的响应，以最适合当前网络条件。

### 3.4 Hierarchical Recurrent Architecture

        在本节中，我们将介绍RL Agent中使用的模型体系结构。受[8]中提出的分层多尺度递归神经网络架构（HM-RNN）的启发，我们设计了一个分层递归模型，其中观察者和策略生成器都采用递归神经网络（RNN）作为基本构建块，并与报告和更新通信相连接。第一层表示观察者，它被输入状态𝑠𝑡 并自适应地触发上层。第二层表示输出子策略参数设置的策略生成器𝑎𝑡 = (𝛼𝑡ℎ𝑟 , 𝛼𝑙𝑎𝑡 , 𝛼𝑡𝑜𝑙 , 𝛼𝑙𝑜𝑠𝑠 )。一旦触发，它将从观察者层接收提交的报告，然后，i）生成一个新的参数集𝑎𝑡 ；ii）更新观察者。通过将策略生成器和观察者集成在一起，我们可以执行梯度下降以协作学习两个模块中的模型权重。

<img title="" src="file:///F:/qiqiblog.github.io/images/2022.11.21-Spine-An-Efficient-DRL_based-Congestion-Control-with-Ultra_low/The%20hierarchical%20recurrent%20neural%20network%20architecture%20with%20different%20timescales.png" alt="avatar" data-align="center" width="397">

## 4 Analysis

        在本节中，我们对Spine进行了理论过时性分析，并假设控制子策略会使模型对控制间隔不太敏感。然后，我们介绍观察者模块如何减少控制系统的开销。

        **对性能和控制间隔之间的权衡进行建模**。我们的算法工作在固定的监视间隔（即，观察者、agent的工作频率是1/T），在每个间隔中检测网络状态，并更新最适合的子策略。我们假设子策略将随着网络环境的变化而过时，我们将其定义为policy drift event。该事件发生后，由于子策略过时，性能将下降，直到Spine在下一个MI中更新子策略。为了简化分析，我们假设子策略以二进制模式工作：它要么工作“好”，要么工作“坏”。因此，子策略过时（或更新）的所有时间段在性能方面都受到同等对待。我们将子策略运行良好时的有效时间与总时间的比率称为有效时间比（effective time ratio，ETR），并在本节中使用它来衡量CC算法的有效性。

        定理1：假设两个policy drift event之间的时间符合参数为的指数分布𝜆。MI长度固定为T的Spine的有效时间比是$\frac{1}{k}(1-e^{-k})$，其中$k=\lambda T$表示在一个MI期间发生的policy drift event的预期数量。

        分布参数𝜆 定义策略漂移事件频率以及子策略过时的速度。这既取决于网络环境的动态性，也取决于子政策结构。长期有效的子策略应具有较低的𝜆 值。基于该定理，给出了有效时间比如何根据MI长度𝑇和𝜆变化。结果直观地表明，当策略漂移事件频率较低时，例如𝜆 = 1，这意味着子策略平均工作一秒钟。如果我们考虑调整𝑐𝑤𝑛𝑑 作为一种子策略，它将导致更大的策略漂移事件频率，因为发送速率需要快速更新以响应可用的动态带宽。如何设计子策略的参数以最小化𝜆 （即，最大化有效期）是我们希望在未来研究的一个有趣的话题。

        **观察者分析**。根据定理1中的假设可知，如果$\frac{cost_w}{cost_p}=e^{-k}$，那么又或者没有观察者的开销将会一致，其中$cost_w$表示观察者（watcher）的开销，$cost_p$表示策略生成器（policy generator）的开销。

        因此，当k相对于我们的策略很小的时候，观察者的消耗能够轻易的小于给策略生成器带来开销收益（即小于$e^{-k}cost_p$）。例如，当𝑇 = 200𝑚𝑠 和𝜆 = 1时，只要观察者的开销小于$e^{-k}\approx81.9\%$的策略生成器的开销，那么Spine就能从中获益。

## 5 Training Algorithm

        为了设计基于DRL的CC算法，我们首先将CC问题表述为强化学习问题。在第t个MI，流和agent通过下面方式交替与网络环境交互：将数据包统计信息作为state$s_t\in \mathcal{S}$观察网络环境，然后基于agent策略生成新的子策略$a_t\in \mathcal{A}$，子策略通过在下一个MI期间调整发送速率来响应分组级信号，并且流将获得奖励𝑟𝑡 基于奖励函数和新收集的统计数据作为下一状态𝑠𝑡+1。尽管公式假设代理为每个MI输出子策略，但它不会与我们的自适应更新策略冲突，我们可以在观察者未被触发的时间间隔内直接重用旧的子策略。agent的目标是最大化期望累计reward，$\mathcal{J}=\mathbb{E}(\sum_{t=0}^{T}\gamma^{t}r_{t})$，其中$\gamma$是折扣因子，用于agent更关注比较近的未来的reward。

        我们采用深度确定性策略梯度（Deep Deterministic policy gradient, DDPG），一种著名的无策略RL算法模型来学习子策略。在训练期间，RL代理更新我们的分层递归神经网络的模型参数，以调整从分组统计到子策略的映射，从而最大化收集的奖励。Spine训练算法的关键特征如下。

**Stored hidden state and burn-in steps**。Spine采用递归模型作为策略模型，它接收并生成隐藏状态以编码历史信息。然而，传统的RL训练方法只存储交互信息，而忽略了过程中生成的隐藏状态，这可能导致历史信息丢失和训练不稳定。我们采用[21]中提出的技巧来解决这个问题。首先，我们将重复的隐藏状态存储在收集的轨迹中，并在训练期间使用它来初始化策略模型。第二，当对交互序列进行采样以进行训练时，我们也会在开始时提取序列的额外部分（老化步骤），这仅用于转发阶段，以在采样序列的开始处产生良好的隐藏状态。（没有太看懂）

**Probabilistic trigger**。在训练观察者的触发单元时存在两个问题：i）它是不可微的。求trigger的函数的导数几乎处处为零，因此梯度反向传播无法进行；ii）由于探索在强化学习中发挥着重要作用，以收集丰富的经验，因此当未触发时，确定性触发单元阻止代理探索更多样化的子策略决策（例如，我可以用更好的子策略挑战现状吗？）。因此，我们通过在训练期间用概率单元替换trigger的函数中的触发单元，将噪声注入到单元中。（具体看论文，公式太复杂了，懒得打）

## 6、Implementation

**模型结构**。我们使用Pytorch[36]构建§3.4中的分层策略模型，其中LSTM用作递归模型的构建块。观察者和策略生成器中的LSTM层分别由64和128维隐藏状态向量组成。在策略生成器中，我们提供LSTM层隐藏状态$h_t^p$到MLP层和tanh层得到输出动作$a_t$。训练期间使用的critic模型也采用了具有128维隐藏状态向量的单个LSTM层。对于CC执行器，我们在Linux内核TCP堆栈中实现了一个拥塞控制模块，该模块从RL代理接收控制参数并执行ACK级拥塞控制。受CCP[34]的启发，CC执行器和用户空间RL模型通过网络链接[37]进行通信。

**训练**。我们的训练基于一个通用的RL培训框架DI引擎[11]，它支持各种DRL算法和定制的环境和策略。我们基于Pantheon[44]构建了仿真拥塞控制训练环境，其中Mahimahi[35]用于仿真各种网络条件。训练环境的设置范围如表3所示。我们还添加了随机数量的Cubic流作为背景流量。我们使用8名演员并行收集培训经验。整个训练超参数集表6中给出。我们在Linux服务器上训练和评估Spine，该服务器具有80个CPU核、256GB RAM，并配备NVIDIA GeForce RTX 3090 GPU。

<img title="" src="file:///F:/qiqiblog.github.io/images/2022.11.21-Spine-An-Efficient-DRL_based-Congestion-Control-with-Ultra_low/Training%20environment%20parameters.png" alt="avatar" data-align="center" width="431">

<img title="" src="file:///F:/qiqiblog.github.io/images/2022.11.21-Spine-An-Efficient-DRL_based-Congestion-Control-with-Ultra_low/Training%20hyperparameters%20in%20Spine.png" alt="avatar" data-align="center" width="252">

## 7、Evaluation

        在本节中，我们通过仿真和真实的测试台实验来评估Spine的性能。在§7.1中，我们展示了Spine如何在低DRL模型推断频率下保持高性能，因为它对监控间隔不敏感。在§7.2中，我们证明了Spine在广泛的网络条件下（包括动态变化的网络条件）实现了一致的高性能。为了更好地理解Spine的控制逻辑，我们将在§7.3中检查Spine如何更新其子策略。我们在§7.4中评估了Spine的收敛性。最后，我们检查了观察者模块带来的改进，并在§7.5中探讨了Spine的更多可能性。

### 7.1 Monitor Interval Insensitivity
