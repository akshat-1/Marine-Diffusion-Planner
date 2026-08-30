# Complete Guide: CPU Cluster Setup & Distributed PyTorch Training (HDP Model)

This guide provides step-by-step instructions to set up a cluster of bare-metal or virtual CPU nodes from scratch (with no pre-installed Python or packages) and run distributed training using **PyTorch DDP (`gloo` backend)**.

---

## 1. Cluster Prerequisites & Network Setup

Suppose you have **Node 0 (Master)** and **Node 1, Node 2... (Workers)**.

### Step 1.1: Determine IP Addresses
Find the private IP addresses of all nodes:
```bash
# On each node:
ip a
```
Example IP layout:
- **Node 0 (Master)**: `192.168.1.100`
- **Node 1 (Worker 1)**: `192.168.1.101`
- **Node 2 (Worker 2)**: `192.168.1.102`

### Step 1.2: Set Up `/etc/hosts` (On ALL Nodes)
Edit `/etc/hosts` on every node so they can communicate by hostname:
```bash
sudo nano /etc/hosts
```
Add:
```text
192.168.1.100 node0
192.168.1.101 node1
192.168.1.102 node2
```

### Step 1.3: Set Up Passwordless SSH (From Master to All Nodes)
On **Node 0 (Master)**:
```bash
# Generate SSH key (press Enter for default path and no passphrase)
ssh-keygen -t ed25519 -N ""

# Copy SSH public key to all worker nodes
ssh-copy-id akshat@node0
ssh-copy-id akshat@node1
ssh-copy-id akshat@node2
```
Test SSH access without password:
```bash
ssh node1 "hostname"
ssh node2 "hostname"
```

---

## 2. Installing Python & Environment on ALL Nodes

Run these steps on **EVERY node in the cluster** (or write a script using SSH).

### Step 2.1: Update System Packages & Install Build Tools

#### For Ubuntu / Debian:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential gcc g++ make git curl rsync \
                    python3 python3-pip python3-venv python3-dev \
                    libxml2-dev libxslt1-dev libgeos-dev libproj-dev
```

#### For Arch Linux:
```bash
sudo pacman -Syu --noconfirm base-devel git curl rsync python python-pip geos proj
```

#### For RHEL / Fedora / CentOS:
```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y python3 python3-devel python3-pip git curl rsync geos-devel proj-devel
```

### Step 2.2: Create Python Virtual Environment (On ALL Nodes)

On every node, create identical project directories:
```bash
mkdir -p /home/akshat/Documents/Diffusion
cd /home/akshat/Documents/Diffusion

# Create virtualenv
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2.3: Install PyTorch CPU & Dependencies

Install CPU-optimized PyTorch and all project dependencies:

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU-only build, significantly faster on CPU clusters)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install commonocean-io commonroad-io geopandas shapely pyproj zstandard pandas numpy scipy pymupdf
```

Verify PyTorch CPU installation:
```bash
python -c "import torch; print('Torch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
# Output should be: Torch Version: 2.x.x+cpu | CUDA Available: False
```

---

## 3. Synchronizing Code & Scenario Data

All nodes **must have access to the exact same codebase and scenario files**.

### Option A: Shared Storage via NFS (Recommended for large datasets)
If you set up an NFS share at `/mnt/nfs_scenarios`, mount it on all nodes:
```bash
# On Master (NFS Server): export /run/media/akshat/Akshat_USB/all_scenerios
# On Workers (NFS Clients): mount master:/run/media/akshat/Akshat_USB/all_scenerios /mnt/nfs_scenarios
```

### Option B: Synchronize Scenarios via `rsync`
If nodes have local SSDs, sync scenarios from Master to all Workers:
```bash
# From Master node:
rsync -avzP /run/media/akshat/Akshat_USB/all_scenerios/ akshat@node1:/home/akshat/all_scenerios/
rsync -avzP /run/media/akshat/Akshat_USB/all_scenerios/ akshat@node2:/home/akshat/all_scenerios/
```

### Sync Code Repository:
```bash
# From Master node:
rsync -avzP --exclude '.venv' /home/akshat/Documents/Diffusion/ akshat@node1:/home/akshat/Documents/Diffusion/
rsync -avzP --exclude '.venv' /home/akshat/Documents/Diffusion/ akshat@node2:/home/akshat/Documents/Diffusion/
```

---

## 4. Key Cluster Configuration Parameters

When running PyTorch Distributed Data Parallel (DDP) across CPU nodes, the following parameters are used:

| Parameter | Environment Variable | Example Value | Description |
|-----------|----------------------|---------------|-------------|
| **Master Address** | `MASTER_ADDR` | `192.168.1.100` | IP of Node 0 (Master) |
| **Master Port** | `MASTER_PORT` | `29500` | Free TCP port on Master node |
| **Nodes Count** | `--nnodes` | `3` | Total number of physical nodes in cluster |
| **Node Rank** | `--node_rank` | `0` (for master), `1`, `2` | Index of the current node |
| **Processes Per Node** | `--nproc_per_node` | `4` | Number of worker processes per node (e.g. 4 CPU worker processes) |
| **Total World Size** | `WORLD_SIZE` | `3 × 4 = 12` | Total processes across whole cluster |
| **Backend** | `backend` | `"gloo"` | PyTorch inter-node communication engine for CPUs |
| **CPU Threads Per Proc** | `OMP_NUM_THREADS` | `4` | Prevents CPU thread oversubscription |

### ⚠️ CPU Oversubscription Prevention Rule:
If your CPU has 16 physical cores and you set `--nproc_per_node=4`:
Set `OMP_NUM_THREADS = 16 / 4 = 4` to prevent threads from fighting over CPU cores!

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

---

## 5. Step-by-Step Distributed Execution Commands

PyTorch provides `torchrun` (standalone launcher) to coordinate distributed execution.

### Example Setup:
- **Node 0 (Master, IP: 192.168.1.100)**
- **Node 1 (Worker 1, IP: 192.168.1.101)**
- 4 processes per node (8 total world size)

---

### Step 5.1: Execute `torchrun` on Node 0 (Master)

Open a terminal on **Node 0**:
```bash
cd /home/akshat/Documents/Diffusion
source .venv/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export GLOO_SOCKET_IFNAME="eth0"  # Replace eth0 with your network interface name from 'ip a'

torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train.py
```

---

### Step 5.2: Execute `torchrun` on Node 1 (Worker 1)

Open a terminal on **Node 1**:
```bash
cd /home/akshat/Documents/Diffusion
source .venv/bin/activate

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export GLOO_SOCKET_IFNAME="eth0"

torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train.py
```

---

## 6. Automated One-Command Cluster Launcher Script

Instead of manually logging into each node, use this Master launch script on **Node 0**.

Create `launch_cpu_cluster.sh` on **Node 0**:

```bash
#!/usr/bin/env bash
# ==============================================================================
# PyTorch CPU Cluster Launch Script (HDP Base Model Training)
# ==============================================================================

MASTER_ADDR="192.168.1.100"
MASTER_PORT="29500"
NODES=("192.168.1.100" "192.168.1.101")  # List all node IPs
NPROC_PER_NODE=4                         # Processes per node
OMP_THREADS=4                            # CPU threads per process
IFNAME="eth0"                            # Network interface name from 'ip a'
PROJECT_DIR="/home/akshat/Documents/Diffusion"

NNODES=${#NODES[@]}
echo "🚀 Launching PyTorch DDP CPU Cluster Training on $NNODES nodes..."
echo "Master IP: $MASTER_ADDR:$MASTER_PORT | Processes per node: $NPROC_PER_NODE"

for RANK in "${!NODES[@]}"; do
    NODE_IP="${NODES[$RANK]}"
    echo "Starting Rank $RANK on $NODE_IP..."

    CMD="cd $PROJECT_DIR && \
         source .venv/bin/activate && \
         export OMP_NUM_THREADS=$OMP_THREADS && \
         export MKL_NUM_THREADS=$OMP_THREADS && \
         export GLOO_SOCKET_IFNAME=$IFNAME && \
         torchrun \
             --nproc_per_node=$NPROC_PER_NODE \
             --nnodes=$NNODES \
             --node_rank=$RANK \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             train.py"

    if [ "$RANK" -eq 0 ]; then
        # Run master in foreground
        eval "$CMD"
    else
        # Run workers remotely via SSH
        ssh "akshat@$NODE_IP" "$CMD" &
    fi
done

wait
echo "Cluster training complete!"
```

Make script executable:
```bash
chmod +x launch_cpu_cluster.sh
./launch_cpu_cluster.sh
```

---

## 7. Parameters & Code Checklist

Before starting cluster training, review these configuration parameters in `train.py`:

1. **Scenario Directory Path**:
   Ensure `SCENARIO_DIR` in `train.py` matches the path where scenario XMLs are stored on each node:
   ```python
   SCENARIO_DIR = "/home/akshat/all_scenerios"  # Or local node path / NFS path
   ```

2. **Effective Batch Size**:
   ```
   Global Batch Size = BATCH_SIZE × NPROC_PER_NODE × NNODES
   ```
   Example: `BATCH_SIZE = 32`, `NPROC_PER_NODE = 4`, `NNODES = 2` $\implies$ Global Batch Size = **256**.

3. **Learning Rate Adjustment**:
   If scaling global batch size by $K$, scale learning rate linearly:
   $$\text{lr} = 5\times 10^{-4} \times \frac{\text{Global Batch Size}}{32}$$

4. **Monitoring & Outputs**:
   - Checkpoints are saved under `weight/checkpoint_epoch_X.pt` on Node 0.
   - Logs are printed only on Rank 0 (`if rank == 0`).

---

## 8. Summary Checklist

- [ ] Network configured (`/etc/hosts` updated on all nodes)
- [ ] Passwordless SSH set up between Node 0 and worker nodes
- [ ] Build dependencies & Python 3 installed on all nodes
- [ ] Virtualenv created and PyTorch CPU installed on all nodes
- [ ] Code & scenarios synced to all nodes (or mounted via NFS)
- [ ] `OMP_NUM_THREADS` set to avoid CPU thread oversubscription
- [ ] `torchrun` launched on all nodes pointing to `MASTER_ADDR:MASTER_PORT`
