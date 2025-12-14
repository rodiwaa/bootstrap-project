QUERIES_BLOG = {
  "REWARD_HACKING": "What does Lilian Weng say about types of reward hacking?",
  "HALLUCINATION": "What does Lilian Weng say about causes of hallucination?",
  "DIFFUSION": "What different architecture does Lilian Weng talk about for diffusion models?",
}

QUERIES_ROHIT = {
  "WHO": "Who is rohit diwakar?",
  "WHAT": "what does he do?",
  # "PROJECT": "What projects has he worked on",
  "PROJECT_AI": "What projects has he worked on in AI?",
  "WORK_EXP": "What is his work experience so far?",
  "PROJECT": "What projects has he worked on so far?",
  "WORK_TIMELINE": "Give timelines of his work experience so far"
}

# Golden answers provide the ground-truth snippets used when logging Opik metrics.
# was using this for evaluation testing. not relevant to project now.
EXPECTED_ANSWERS = {
  "REWARD_HACKING": (
    "Lilian Weng splits reward hacking into two broad classes: misspecified goals or "
    "environments where the proxy reward fails to reflect the task, and reward tampering "
    "(a.k.a. wireheading) where the agent manipulates or shortcuts the reward channel "
    "instead of solving the task."
  ),
  "HALLUCINATION": (
    "She attributes hallucinations to knowledge gaps and brittle reasoning (insufficient or "
    "outdated training data), to decoding dynamics such as high temperature sampling, and to "
    "misaligned incentives where the model prioritizes fluent answers over factual grounding."
  ),
}
