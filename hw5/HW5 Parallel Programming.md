---
title: HW5 Parallel Programming

---

# HW5 Parallel Programming
**學號:** 112062532 　　**姓名:** 温佩旻
## 1. Overview
>In conjunction with the UCP architecture mentioned in the lecture, please read [ucp_hello_world.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/pp2024/examples/ucp_hello_world.c)
1. Identify how UCP Objects (`ucp_context`, `ucp_worker`, `ucp_ep`) interact through the API, including at least the following functions:
    * `ucp_init`
        `ucp_context_h` 是負責管理 UCP application 的global context的物件指標，透過call [ucp_init](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/ce5c5ee4b70a88ce7c15d2fe8acff2131a44aa4a/examples/ucp_hello_world.c#L575) API，並傳入 [ucp_params_t](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/ce5c5ee4b70a88ce7c15d2fe8acff2131a44aa4a/examples/ucp_hello_world.c#L563C5-L563C15)，可以創建並初始化 `ucp_context_h`。
    * `ucp_worker_create`
        透過call [ucp_worker_create](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/ce5c5ee4b70a88ce7c15d2fe8acff2131a44aa4a/examples/ucp_hello_world.c#L587)，可以基於已透過 [ucp_init](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/ce5c5ee4b70a88ce7c15d2fe8acff2131a44aa4a/examples/ucp_hello_world.c#L575) 初始化的 `ucp_context`，以及設定好的 `worker_params`（用來指定 thread mode 的參數），來創建並初始化 `ucp_worker` object。
    * `ucp_ep_create`
        `ucp_ep` 代表與遠端節點的connection，透過 [ep_params](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/ce5c5ee4b70a88ce7c15d2fe8acff2131a44aa4a/examples/ucp_hello_world.c#L444C1-L444C1)（包含client的 UCX address與其他資訊）以及已經初始化的 `ucp_worker`，來初始化 `ucp_ep` object，建立與遠端節點的連接。
        
    完成上述三個API的初始化後，application就可以利用`ucp_worker`來和不同的client communication，其中一個`ucp_worker`可以看成是負責處理通訊的一個thread，而`ucp_ep`表示的是該thread和其他process的connection。
    
2. UCX abstracts communication into three layers as below. Please provide a diagram illustrating the architectural design of UCX.
    * `ucp_context`
        提供application的global context，管理 UCX 的初始化和資源配置，包括記憶體註冊和底層硬體的資訊，每個application只需創建一個 `ucp_context`，負責生成 `ucp_worker`。
    * `ucp_worker`
        代表communication的執行單元，負責管理與傳輸相關的資源，通常一個thread對應到一個`ucp_worker`，每個worker可以有多個 `ucp_ep`，每個worker只能用一種transport方法，要處理怎麼對endpoint去做傳輸。
    * `ucp_ep`
        UCP endpoint，表示從本地 worker 到遠端 worker 的連接，當每個worker需要和遠端進行communication時，就會建立一個 `ucp_ep`，每個worker可以有很多個 `ucp_ep`，和不同host連接，主要用於發送和接收data，並由 `ucp_worker_h` 負責建立。
        
![image](https://hackmd.io/_uploads/HyG7HJiEkx.png)

>Please provide detailed example information in the diagram corresponding to the execution of the command `srun -N 2 ./send_recv.out` or `mpiucx --host HostA:1,HostB:1 ./send_recv.out`

* `mpiucx --host HostA:1,HostB:1 ./send_recv.out`
    指定 HostA 和 HostB 各執行 1 個process，HostA 和 HostB 各創建自己的 `ucp_context_h` 和 `ucp_worker_h`，HostA 與 HostB 之間創建對應的 `ucp_ep_h` 來建立connection，通過 `ucp_ep_h`進行點對點的資料交換。當Host A 的 `ucp_worker_h` 使用 `ucp_ep_h` 發送data時，data會經由 RDMA 傳輸到Host B 的 `ucp_worker_h`，由該 worker 處理後上傳至應用層。
    
3. Based on the description in HW5, where do you think the following information is loaded/created?
    * `UCX_TLS`
        TLS是所有目前系統能夠選擇的transport 方法，所以我認為應該是 `ucp_context` 在初始化階段的時候load進來的。
    * TLS selected by UCX
        這邊是指實際被選擇的TLS，我認為應該是 `ucp_worker` 處理的，因為每個 `ucp_worker` 可以選用不同的transport 方法。


## 2. Implementation
1. Which files did you modify, and where did you choose to print Line 1 and Line 2?
    * **print Line 1:**
        * 修改 [ucp_worker.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c) 這份檔案。
        
            |![image](https://hackmd.io/_uploads/r1AXsMiVJl.png)| 
            |:------------------:|
            |圖 1|
            
            我在[ucp_worker.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c) 中的 [ucp_worker_get_ep_config](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c#L2033)這個function中新增 `ucp_config_read` 來取得UCX_TLS相關資訊，接著會 call `ucp_config_print` ，透過 [`parser.c`](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucs/config/parser.c) 的 [ucs_config_parser_print_opts](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucs/config/parser.c#L1783) 這個function將取得的TLS資料印出。
        
        * 修改 [parser.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucs/config/parser.c) 這份檔案。
            |![image](https://hackmd.io/_uploads/B1lldfo4kl.png)|
            |:------------------:|
            |圖 2|
            
            在[ucs_config_parser_print_opts](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucs/config/parser.c#L1783) 中完成上圖的TODO，印出UCX_TLS
        
        
        * 修改 [types.h](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucs/config/types.h) 這份檔案。

            |![image](https://hackmd.io/_uploads/Sk833fjVyg.png)|
            |:------------------:|
            |圖 3|
        
            由於在[ucp_worker.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c) 和 [parser.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucs/config/parser.c) 都會用到 `UCS_CONFIG_PRINT_TLS` ，所以需要在 [types.h](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucs/config/types.h) 中的 [ucs_config_print_flags_t](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucs/config/types.h#L88) 加入 `UCS_CONFIG_PRINT_TLS` ，如上圖所示。
    * **print Line 2:**
        修改[ucp_worker.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c)這份檔案。
        從助教的提示中可以觀察到在 run `mpiucx -x UCX_LOG_LEVEL=info -np 2 ./mpi_hello.out`的時候， `ucp_worker.c` 的[ucp_worker_print_used_tls](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c#L1855) function 中就有印出TLS selected by UCX的資訊，所以在print Line 2的部分就直接在 [`ucs_info("%s", ucs_string_buffer_cstr(&strb));`](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c#L1855)的後面加上一行`printf("%s\n", ucs_string_buffer_cstr(&strb));` 就能印出SPEC指定的line 2資訊。
    
2. How do the functions in these files call each other? Why is it designed this way?
    在 `ucp_worker.c` 的 `ucp_worker_get_ep_config` 中會先call `ucp_config_read` (圖1的2106行) 來取得UCS_TLS的資訊，接著會 call `ucp_config_print` (圖1的2107行) ，`ucp_config_print` 裡面會再call `ucs_config_parser_print_opts` (圖2)，將TLS的資訊印出，接著 `ucp_worker.c` 的 `ucp_worker_get_ep_config` 會再call [`ucp_worker_print_used_tls`](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c#L2104) (圖1的2108行) ，將目前所使用的TLS資訊印出。

    由於前面說到的，透過助教的提示中可以觀察到在 run `mpiucx -x UCX_LOG_LEVEL=info -np 2 ./mpi_hello.out`的時候，`ucp_worker.c` 的[ucp_worker_print_used_tls](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c#L1855) 會印出目前所使用的TLS的資訊，因此只需要用同一個function所獲的的資訊直接印出Line 2就好。接著透過trace code發現 `ucp_worker_get_ep_config` 會 call `ucp_worker_print_used_tls` ，因此我在 call `ucp_worker_print_used_tls` 之前將全部可用的TLS的資訊印出，完成Line 1的實作。

3. Observe when Line 1 and 2 are printed during the call of which UCP API?
    從上面的敘述中可以發現我的Line 1和Line 2都寫在[ucp_worker_get_ep_config](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_ep.c#L790) 這個function當中，從這個function回朔回去，可以觀察到最初call的UCP API 是 [ucp_ep_create](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_ep.c#L1176) 這個API，在 `ucp_ep_create` 中會call [ucp_ep_create_to_sock_addr](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_ep.c#L1186)，接著再從 `ucp_ep_create_to_sock_addr` 中 call [ucp_ep_init_create_wireup](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_ep.c#L876)，最後在 `ucp_ep_init_create_wireup` function 裡面call [ucp_worker_get_ep_config](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_ep.c#L790)，然後就可以印出Line 1和Line 2。

4. Does it match your expectations for questions 1-3? Why?
    * `UCX_TLS`
        在前面我推測這是會在 `ucp_context` 中被 load 進來，因為 `UCX_TLS` 是 global資訊，trace code 後也確實是如此，在 [ucp_init](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/examples/ucp_hello_world.c#L575) function 中會將透過 `ucp_config_read` 得到的UCX_TLS讀到 `ucp_context` 。
不過實作中因為在 `ucp_worker_get_ep_config` 才會取得真正被取用的TLS，所以我將 `ucp_config_read` 及 UCX_TLS 印出的地方寫在 `ucp_worker_get_ep_config` 的 `ucp_worker_print_used_tls` 之前。

    * TLS selected by UCX
        在前面我推測是 `ucp_worker` 處理的，但 trace code 後發現 `ucp_worker` 是提供了所有可用的 TLS（例如 TCP、RDMA等），`ucp_ep` 才從這些可用的 TLS 中選擇最終要用的協議。從 [ucp_worker_get_ep_config](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_worker.c#L2033) 傳入的參數 key ，查找 `worker->ep_config` 中是否已有相同的 key，如果找到，代表已經配置過了，用`ep_cfg_index` 獲得現有配置的index，則最終協議已經由之前的初始化決定，如果沒有找到則透過 [ucp_ep_config_init](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/84e459e73df4f02aecd044c44e4584d88f4b9b0e/src/ucp/core/ucp_ep.c#L2465) 來創建新配置，根據 key 初始化每個Lane的協議，並存入`ep_config`，作為該端點的最終協議。
        

5. In implementing the features, we see variables like lanes, tl_rsc, tl_name, tl_device, bitmap, iface, etc., used to store different Layer's protocol information. Please explain what information each of them stores.
    * **lanes:** 每個 Lane 對應到 UCX 中的一個傳輸層協議（如: TCP、RDMA）以及具體的設備（如 eth0 或 mlx5_0），用來決定數據傳輸中使用的傳輸協議和設備，例如: lanes[0] 對應 TCP，lanes[1] 對應 RDMA。
    * **tl_rsc:** transport layer的相關資訊，包括tl_name、tl_device。
    * **tl_name:** transport layer的名字，像是ud_verbs、tcp、rdma。
    * **tl_device:** transport layer可以使用的具體device名稱，像是eth0、mlx5_0。
    * **bitmap:** 表示可用資源或 Lane 的狀態，每一bit對應於一個資源（如 Lane 或傳輸協議）的啟用或禁用狀態，能用來快速檢查哪些 Lane 或資源是可用的。
    * **iface:** 用來管理數據的發送和接收，每個 iface 由 Lane 綁定到特定的Transport Layer和device，例如iface 可能是 TCP 的 socket entity 或 RDMA 的 QP（Queue Pair）。

## 3. Optimize System
1. Below are the current configurations for OpenMPI and UCX in the system. Based on your learning, what methods can you use to optimize single-node performance by setting UCX environment variables?

    ```
    -------------------------------------------------------------------
    /opt/modulefiles/openmpi/ucx-pp:

    module-whatis   {OpenMPI 4.1.6}
    conflict        mpi
    module          load ucx/1.15.0
    prepend-path    PATH /opt/openmpi-4.1.6/bin
    prepend-path    LD_LIBRARY_PATH /opt/openmpi-4.1.6/lib
    prepend-path    MANPATH /opt/openmpi-4.1.6/share/man
    prepend-path    CPATH /opt/openmpi-4.1.6/include
    setenv          UCX_TLS ud_verbs
    setenv          UCX_NET_DEVICES ibp3s0:1
    -------------------------------------------------------------------
    ```
    可以看到apollo預設是將UCX_TLS設定為 ud_verbs，ud_verbs 會使用RDMA，而RDMA需要到 remote 將memory copy過來，進而產生 memory copy 的 overhead，這邊因為是single-node，我認為可以考慮設成shared memory，這樣可以減少 memory copy 的 overhead，這樣應該會比走網路的速度要來的快。

2. Please use the following commands to test different data sizes for latency and bandwidth, to verify your ideas:
    ```bash
    module load openmpi/ucx-pp
    mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_latency
    mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/osu_bw
    ```
    圖4、圖5是使用UCX預設的 `UCX_TLS=ud_verbs` 去測試 osu_latency(圖4) 及 osu_bandwidth(圖5) 的結果。圖6、圖7則是將UCX_TLS設定為all，讓UCX去選擇最好的TLS，可以看到UCX都是選擇shared memory，等同於`UCX_TLS=sm`，這邊可以觀察到使用 `UCX_TLS=all` 能讓latency減少2~3倍，也讓 bandwidth 增加3倍多，根據這些結果表示了在 single-node 上採用 shared memory 確實可有效減少 latency 及增加 bandwidth。
    |![image](https://hackmd.io/_uploads/rJPyc83Vkl.png)|![image](https://hackmd.io/_uploads/B1H6q8hV1e.png)|
    |:------------------:|:------------------:|
    |圖 4|圖 5|
    
    |![image](https://hackmd.io/_uploads/ryYj5IhN1g.png)|![image](https://hackmd.io/_uploads/Hys72L3NJl.png)|
    |:------------------:|:------------------:|
    |圖 6|圖 7|

    |![image](https://hackmd.io/_uploads/Hkm6TU2V1e.png)|
    |:------------------:|
    |圖 8 `UCX_TLS=ud_verbs` 和 `UCX_TLS=all` latency 比較|
    
    |![image](https://hackmd.io/_uploads/ry8bC82NJg.png)|
    |:------------------:|
    |圖 9 `UCX_TLS=ud_verbs` 和 `UCX_TLS=all` bandwidth 比較|
      
    接著我觀察到 `UCX_TLS=sm` 時同時啟用了`cma/memory` 和 `sysv/memory`，所以測試了`UCX_TLS=sm` 和 `UCX_TLS=sysv` latency 和 bandwidth 的比較，將結果用圖10、圖11兩張表格呈現，可以發現在 single-node 上，數據量越大， `UCX_TLS=sysv` 在 latency 和 bandwidth 上的表現相對於 `UCX_TLS=sm` 的表現就越好，因此除了設定 `UCX_TLS=all` 能提高performance，設定`UCX_TLS=sysv` 在資料量大的時候表現會更不錯。
    
    |![image](https://hackmd.io/_uploads/rkG-Nv2N1e.png)|
    |:------------------:|
    |圖 10 `UCX_TLS=sm` 和 `UCX_TLS=sysv` latency 比較|
    
    |![image](https://hackmd.io/_uploads/S1ATSwnEyx.png)|
    |:------------------:|
    |圖 11 `UCX_TLS=sm` 和 `UCX_TLS=sysv` bandwidth 比較|
    
3. Please create a chart to illustrate the impact of different parameter options on various data sizes and the effects of different testsuite.
    因為 `UCX_TLS=all` 選出來的結果基本上就是 `UCX_TLS=sm` ，所以這邊我比較 `UCX_TLS=sm` 、 `UCX_TLS=sysv` 、`UCX_TLS=ud_verbs` 三種設定隨著 size 不同，latency 和 bandwidth 的相關性為何。
    
    |![image](https://hackmd.io/_uploads/BJWCYv2E1g.png)|
    |:------------------:|
    |圖 12 `UCX_TLS=sm`、`UCX_TLS=sysv`、`UCX_TLS=ud_verbs` latency 比較|
    
    |![image](https://hackmd.io/_uploads/S1d6_PnNJe.png)|
    |:------------------:|
    |圖 13 `UCX_TLS=sm`、`UCX_TLS=sysv`、`UCX_TLS=ud_verbs` bandwidth 比較|


4. Based on the chart, explain the impact of different TLS implementations and hypothesize the possible reasons (references required).
    從圖12可以發現，`UCX_TLS=ud_verbs` 隨著數據大小增加，latency會急劇上升，尤其是數據量愈大攀升的幅度越明顯，相較之下，`UCX_TLS=sm` 和 `UCX_TLS=sysv` 就相對穩定一些，這樣的原因是因為 ud_verbs 是基於Unreliable Datagram 協議，數據傳輸需要經過網路(如RDMA網卡)，在大數據傳輸下，網路傳輸延遲、網路資源競爭等等的開銷會更明顯，所以相比於shared memory就會慢的許多。
    同時也可以發現在大數據量下，latency_sm 比 latency_sysv 略高，顯示 sysv 在大數據下的延遲更穩定一些，這可能是因為 `UCX_TLS=sm` 同時啟用了多種shared memory機制，如 `cma/memory` 和 `sysv/memory`，在大數據的傳輸時，CMA 可能會觸發頻繁的context switch，導致額外的overhead，進而使 latency_sm 略高於 latency_sysv，而在 `UCX_TLS=sysv` 的模式下，專注於 System V shared memory，所以memory可能更容易被系統優化，能避免額外的context switch，所以在大數據的情況下 `UCX_TLS=sysv` 表現較好。
    從圖13可以發現到 `UCX_TLS=ud_verbs` 隨著數據量提高 bandwidth 上升的幅度並不明顯，可以和上述latency結果呼應，由於延遲過高，單位時間內能夠處理的數據量就會受到限制，原因也是因為網路層的限制，加上不保證數據順序和可靠性，在大數據量下會增加數據包重傳的可能性，更進一步降低有效bandwidth。
    而`UCX_TLS=sm` 則是看起來波動比較明顯，這可能是因為上面提到的， UCX 在 `sm` 模式下會去嘗試多種 shared memory 機制，在不同數據量會切換不同機制，導致性能不穩定，另外可以觀察到 `UCX_TLS=sysv` 在大概中等數據量的時候 bandwidth 就已經明顯高於另外兩個了，顯示出sysv 協議在大數據傳輸時表現較穩定。
    
### Advanced Challenge: Multi-Node Testing
This challenge involves testing the performance across multiple nodes. You can accomplish this by utilizing the sbatch script provided below. The task includes creating tables and providing explanations based on your findings. Notably, Writing a comprehensive report on this exercise can earn you up to 5 additional points.

* For information on sbatch, refer to the documentation at [Slurm's sbatch page](https://slurm.schedmd.com/sbatch.html).
* To conduct multi-node testing, use the following command:
```
cd ~/UCX-lsalab/test/
sbatch run.batch
```
圖14是用run.batch來跑 Multi-Node，設定 `UCX_TLS=all` 在 latency 的結果，圖15是single-node和multi-node 的latency 表格，可以觀察到，`UCX_TLS=all` 在 multi-node 環境下執行時，Latency 明顯比 single-node 環境高，這是因為在 Single-Node 的環境下UCX使用的是shm協議，不需要透過網路進行傳輸，而是直接通過 shared memory 來進行傳輸，但在 Multi-Node 的環境下，UCX會選擇 RDMA (rc_verbs) 和 TCP 作為主要的傳輸協議，這需要通過InfiniBand 或 Ethernet TCP/IP 來傳輸數據，也就是在 Multi-Node 還需要網卡與CPU之間的 memory 交換，會產生額外的 CPU overhead。
|![image](https://hackmd.io/_uploads/S1TfkKhE1e.png)|
|:------------------:|
|圖 14 Multi-Node 設定 `UCX_TLS=all` 在osu_latency表現 |

|![image](https://hackmd.io/_uploads/rJIzZF3E1g.png)|
|:------------------:|
|圖 15 Single-Node 和 Multi-Node 設定`UCX_TLS=all` latency 比較|

圖16是用run.batch來跑 Multi-Node，設定 `UCX_TLS=all` 在 bandwidth 的結果，圖17是single-node 和 multi-node 的 bandwidth 表格，這邊呈現的結果大致上可以和上面 latency 的結果相呼應，在所有 size 下，Single-Node 的 bandwidth 都顯著高於 Multi-Node ，主要原因和上述一樣，因為 Multi-Node 會用到網路傳輸，即使網路 bandwidth 有被充分利用，但上限也遠低於 shared memory 傳輸，導致 bandwidth 上升幅度有限。

|![image](https://hackmd.io/_uploads/rJimQY24yg.png)|
|:------------------:|
|圖 16 Multi-Node 設定 `UCX_TLS=all` 在osu_bandwidth 表現 |

|![image](https://hackmd.io/_uploads/H1COXY2NJg.png)|
|:------------------:|
|圖 17 Single-Node 和 Multi-Node 設定`UCX_TLS=all` bandwidth 比較|


## 4. Experience & Conclusion
1. What have you learned from this homework?
透過這次trace UCX code的作業讓我更了解UCX每個layer的功能和傳輸的架構，原本聽老師上課時感覺很抽象的，比較難理解，但透過實際trace code後會有更具體的了解。
這次作業也讓我體會到，要提升程式的執行效率，不是只有不斷優化 GPU code，相反的，底層的硬體架構和通訊傳輸機制可能更加重要。

2. How long did you spend on the assignment?
大約花一個禮拜