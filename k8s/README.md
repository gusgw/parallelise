# Running a conversion cell on Kubernetes

`job.yaml` holds three Kubernetes objects: a ConfigMap (a named set of settings), a Secret
(a named value kept out of ordinary settings, such as a password) and a Job (a task that
runs once to completion). Together they run one
conversion cell, that is one problem on one backend with one agent, and then stop.

Build the image and make it available to the cluster first (`make image`, then push it to
a registry the cluster can reach and change `image:` in the file). Then run
`kubectl apply -f k8s/job.yaml` and follow the log with
`kubectl logs -f job/parallelise-integrate-openmp`.

## What each field means

- **ConfigMap `parallelise-config`** holds the address of the language-model gateway
  (`gateway-url`) and the model name (`model`). They reach the program as the environment
  variables `PARALLELISE_GATEWAY_URL` and `PARALLELISE_MODEL`. The default address,
  `http://localhost:4000/v1`, only works if a gateway runs in the same pod, so change it
  to the address of your own gateway.
- **Secret `parallelise-secret`** holds the optional key for the gateway. It is passed as
  `PARALLELISE_API_KEY`. The reference is marked `optional`, so the Job still starts if
  you delete the Secret.
- **`backoffLimit: 0`** means that a failed pod is not retried. A retry would spend more
  model requests and would hide the first failure.
- **`activeDeadlineSeconds: 3600`** makes Kubernetes stop the Job after one hour whatever
  it is doing. It is the outer limit; the program has its own shorter timeouts.
- **`ttlSecondsAfterFinished`** deletes the finished Job after a day so that it does not
  accumulate. Read the log before then.
- **`restartPolicy: Never`** stops the container being restarted inside the same pod.
- **`args`** are passed to the `parallelise` command. Change `integrate` and `openmp` to
  run another cell. Add `--model NAME` to override the ConfigMap.
- **Pod `securityContext`** requires a non-root user (identifier 10001, the one built into
  the image), uses the container runtime's default system-call filter, and makes mounted
  volumes writable by that user's group. The service account token is not mounted, because
  the program never talks to the cluster.
- **Container `securityContext`** forbids gaining extra privileges, drops every Linux
  capability and makes the root file system read-only. The generated code therefore cannot
  change the installed programs.
- **`resources`** gives the scheduler the request and caps the container at the limit.
  Four processor cores and 4 GiB of memory are enough for the OpenMP and MPI kernels; a
  runaway program is throttled on processor time and killed if it exceeds the memory limit.
- **Volume `work`** is an empty scratch directory mounted at `/work`, limited to 2 GiB.
  It is the only writable place. It holds the agent's work folder, temporary files and the
  run record, and it disappears with the pod. To keep `runs.jsonl`, copy it out with
  `kubectl cp` while the pod still exists, or print it at the end of the run.

## Things this file does not do

It does not request a graphics card, so it cannot run the CUDA backend. For that, build
the `cuda` stage and add a `nvidia.com/gpu: 1` limit to the container. It also does not
provide a model: the gateway must already be reachable from the pod.
