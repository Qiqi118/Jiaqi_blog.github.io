# Dynamic TCP Initial Windows and Congestion Control Schemes Through Reinforcement Learning

# Abstract

        尽管对TCP进行了多年的改进，但其性能仍然不尽如人意。对于以短流为主的服务（如网络搜索和电子商务），TCP存在流启动问题，无法充分利用现代互联网中的可用带宽：TCP从一个保守和静态的初始窗口（initial window，IW，2-4或10）开始，而大多数网络流太短，无法在会话结束前收敛到最佳发送速率。对于以长流量为主的服务（例如，视频流和文件下载），手动和静态配置的拥塞控制（CC）方案可能无法为最新的网络条件提供最佳性能。为了解决这两个挑战，我们提出了TCP-RL，它使用强化学习（RL）技术来动态配置IW和CC，以提高TCP流传输的性能。基于在web服务的服务器侧观察到的最新网络条件，TCP-RL通过基于组的RL为短流动态配置合适的IW，并且通过深度RL为长流动态配置适合的CC方案。我们的大量实验表明，对于短流量，TCP-RL可以将平均传输时间减少约23%；对于长流量，与14个CC方案的性能相比，TCP-RL的性能在288个给定静态网络条件中的约85%排名前5，而在约90%的条件下，与相同网络条件下性能最好的CC方案相比，其性能下降不到12%。

# 1、Introduction

        大多数在线网络服务（如微软、百度）都是基于TCP传输的；TCP性能直接影响用户体验和公司收入。然而，尽管TCP进行了多年的改进，但其性能仍不令人满意。在本文中，我们重点讨论了两个众所周知的TCP性能问题：（1）TCP无法优雅地处理短流，以及（2）拥塞控制（CC）算法的性能仍远不理想。

        在第一个问题中，TCP以缓慢的开始阶段开始传输，以探测可用带宽，然后使用CC算法收敛到最佳发送速率。具体而言，TCP从保守和静态的初始拥塞窗口（IW、2、4或10）[9]开始，然后它试图通过在流传输期间不断探测和调整其拥塞窗口（CWND）来找到流的最佳发送速率。然而，大多数web流都很短，以至于它们可以在一个具有最佳CWND的RTT中完成，但它们不足以在缓慢启动阶段使用多个RTT来探测其最佳CWND。作为一个真实世界的例子，表I显示，对于全球顶级搜索公司百度的移动搜索服务，IW=10，当会话结束时，超过80%的TCP流仍处于缓慢启动阶段；它们没有充分利用可用的带宽。

        上述TCP流启动问题仍然被研究界认为是针对一般TCP环境的开放研究问题。谷歌提议将标准IW从2-4提高到10。但是，对于高速用户（例如，光纤接入）来说，10是否仍然太小，或者对于低速用户（例如远程地区的GPRS接入）来说10是否太大？由于网络条件在空间和时间上都可能高度可变，因此选择最适合所有流的静态IW是不可行的。

        在第二个问题中，高效的CC算法对于数据传输至关重要，尤其是对于以长流量为主的服务，如视频流和大文件下载。在过去的几十年里，随着网络技术和互联网基础设施的快速发展，已经提出了许多CC方案的变体（例如，Tahoe、Reno、Cubic、BBR、PCC Vivac、Copa、Indigo）。然而，Pantheon最近的一项研究表明不同CC方案的性能在各种网络条件下显著变化，并且没有一个CC方案能够在所有网络条件下优于所有其他方案。由于同一web服务的网络条件可以在时间上（跨时间）和空间上（跨不同用户）变化，为了实现长流量的最佳TCP性能，web服务可能需要跨不同时间和/或用户的不同CC方案。然而，通常的做法是，web服务的提供者手动和静态地为web服务配置一个特定的CC方案，并坚持使用它，从而使TCP性能不理想。

        为了解决上述两个问题，我们认为网络服务提供商可以使用强化学习RL方法来动态配置IW和CC，以提高互联网中的网络传输性能。基本上，网络条件决定了IW和CC的理想值，因此配置IW和CC的过程包括在状态（网络条件）和动作（IW，CC）之间建立映射。在这里，我们将上述两个问题视为RL问题，因为RL的基本思想是通过建立环境状态和行动之间的映射来最大化累积奖励的概念。通过动态决定视频会话的服务前端服务器[15]或通过调整ABR算法[16]，将RL应用于互联网视频QoE优化的最新进展启发和鼓励了我们选择RL。        

## 挑战

    1、 如何仅在服务器端测量新的TCP数据？RL方法需要新的数据来计算奖励和状态。传统的web服务服务器不能在无客户端协作的情况下直接测量一些TCP数据。

    2、挑战2：如何将RL方法应用于互联网高度可变和非连续的网络条件？RL的决策是由上下文的静态（但未知）分布决定的。RL方法需要影响决策回报的上下文的连续性[17]，[18]。在我们的案例中，上下文是流的网络条件（即可用的端到端带宽和RTT），但这些条件在不同的时间和用户粒度（即IP或子网）之间是高度可变的。使用哪个RL方法，在哪个RL方法中用户粒度是个问题。

    挑战3：如何从大的决策空间快速搜索最佳TCP IW或CC？RL方法基本上是基于试错的。它们需要巧妙的探索机制，通常只适用于小而有限的决策空间，这可以在少量试验后迅速收敛到最佳决策。然而，IW（范围从2到100以上）和CC（至少14个变体[8]）的搜索空间太大，以至于在暴力搜索能够找到先前网络条件的最佳IW或CC之前，网络条件可能已经改变。

# 贡献

    1、为了解决挑战1，我们修改了Linux内核和Nginx软件，使web服务器能够在没有任何客户端帮助的情况下实时收集和存储数十亿TCP流性能记录（例如，传输时间、吞吐量、丢失率、RTT）（§VI）。

    2、为了解决挑战2中IW配置问题，我们在组粒度上应用了传统的无模型RL方法，因为应该在传输之前配置IW，而此时没有状态信息（网络条件）。其基本思想是应用在线探索开发。由于细粒度用户组（即IP）的数据样本可能太少，无法检测上下文连续性，我们提出了一种自下而上的方法来对具有相同网络特征的用户的流进行分组，以找到既有足够样本又满足RL上下文连续性要求的最细粒度的用户组。与之前的工作相比，TCP-RL利用了来自用户组的更丰富的历史信息来帮助配置合适的IW。

    3、为了解决挑战2中的CC配置问题，TCP-RL在每个流粒度上使用深度RL方法。离线训练一个能够为给定的网络条件选择合适的CC神经网络模型。TCP-RL建议使用深度RL在每个流级别上为不同的网络条件动态配置正确的CC方案，而不是构建新的CC。

    4、为了解决挑战3，对于IW配置问题，我们使用快速决策空间搜索算法改进了RL。基于对TCP性能和IW之间关系的普遍看法，我们提出了一种滑动决策空间方法，该方法可以快速收敛到最佳IW（§IV-a）。对于CC配置问题，我们离线训练的神经网络模型可以直接在线选择正确的CC，而无需搜索。

# 2、Background

## A. The Preliminary of TCP CC

        已有的拥塞控制算法从一个很小的发送速率开始传输，导致很多短流传输结束的时候的传输速率依然很低。另一方面，没有任何一个算法能够在所有网络环境下表现最好。

## B. Initial Window and Short Flows

        对于短流传输来说，传输完成的时间是衡量性能的关键。对于绝大多数是短流的服务来说，大多数流在slow start阶段就已经传输结束，而IW是决定初始发送速率的关键因素。太小的IW会遭受比完成传输所需的更多的RTT；IW过大会导致拥塞甚至昂贵的TCP超时，从而导致高网络传输时间。

## C. Congestion Control and Long Flows

        对于长流，实现高带宽和低RTT是cc的目标，仅仅调整IW对长流的影响不大。现有的算法中没有一种能够在真实互联网中的不同网络条件下始终保持最佳性能。如果我们能够以某种方式为每个特定的网络条件选择合适的CC，则传输性能可以显著提高。

# 3、Core Ideas And System Overview

        对于短流，IW是影响TCP性能的主要因素，并且流在CC生效之前结束传输。对于长流，CC是实现良好性能的关键，而IW几乎没有影响。因此，分别配置IW和配置CC。

    对于短流的IW配置问题，在单个流中学习IW是不可能的，因为在配置IW之前，发送方无法观察网络环境的状态。

为了应对网络条件的可变性，我们提出了一种自下而上的方法，对来自具有相同网络特征（例如，子网、ISP、省份）的客户端（用户）的流进行分组，以找到既有足够样本又满足RL上下文连续性要求的最细粒度的用户组；然后，我们在每个用户组中应用在线RL方法。
