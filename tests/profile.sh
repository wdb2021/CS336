
#步骤1：仅 CUDA + NVTX 追踪
#/mnt/d/Model/cs336/assignment1-basics-main/tests/a_myLM.py

nsys profile \
  --trace=cuda \
  --capture-range=cudaProfilerApi \
  --stats=true \
  --output=model_report_cuda \
  --force-overwrite true \
  python3 a_myLM.py

