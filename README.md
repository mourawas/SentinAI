# SentinAI

## 🧪 Project Setup & Execution Guide

This guide walks you through setting up and running SentinAI on the EPFL RCP server.

---

### 1. 🔧 Setup Your Docker Environment

Use the Docker image with the required libraries located in the `docker/` folder and build it as follows:

```bash
cd docker/

docker build --platform linux/amd64 . --tag registry.rcp.epfl.ch/ee-559-<username>/my-toolbox:v0.1 \
  --build-arg LDAP_GROUPNAME=rcp-runai-course-ee-559_AppGrpU \
  --build-arg LDAP_GID=84650 \
  --build-arg LDAP_USERNAME=<username> \
  --build-arg LDAP_UID=<uid>
```

> 💡 Replace `<username>` with your RCP username and `<uid>` with your user ID.

This will create a custom Docker image with access permissions and necessary libraries, ready to be used on the Run:AI cluster.

---

## 2. 📂 Upload Project & Dataset to the Server

Store the project in two locations:

- Your **home directory** on the server:  
  `/home/<username>/`
- The **shared group directory**:  
  `/mnt/course-ee-559/collaborative/group-<group-number>/`

#### Using `scp` to Copy Files

From your local machine, run the following commands to upload your project:

```bash
# Copy project folder
scp -r practice_3_repository <username>@jumphost.rcp.epfl.ch:/home/<username>/

# Copy dataset folder
scp -r fruit_dataset <username>@jumphost.rcp.epfl.ch:/home/<username>/

# Optional: Copy to shared group directory
scp -r fruit_dataset <username>@jumphost.rcp.epfl.ch:/mnt/course-ee-559/collaborative/group-<group-number>/
```

### 3. 🧪 Test Your Script on an Interactive Node

Before running the full training, test the script interactively with reduced epochs to avoid long GPU usage:

```bash
runai submit \
  --image registry.rcp.epfl.ch/ee-559-<username>/my-toolbox:v0.1 \
  --pvc home:${HOME} -e HOME=${HOME} \
  --interactive -g 1 --attach
```

Then, inside the interactive node, run:

```bash
python3 ~/practice_3_repository/practice_3_simplified.py \
  --dataset_path ~/ \
  --results_path ~/practice_3_repository/results/
```

---

### 4. 🚀 Submit the Full Training Job

Once the script works correctly:

```bash
runai submit \
  --image registry.rcp.epfl.ch/ee-559-<username>/my-toolbox:v0.1 \
  --gpu 1 \
  --pvc home:${HOME} -e HOME=${HOME} \
  --command -- python3 ~/practice_3_repository/practice_3_simplified.py \
  --dataset_path ~/ \
  --results_path ~/practice_3_repository/results/
```

---

## 5. 🪵 Check Job Logs & Monitor

If the job fails or you want to monitor it:

```bash
runai logs <job-name>
```

Track job status here:  
🔗 [https://rcpepfl.run.ai/](https://rcpepfl.run.ai/)

---
