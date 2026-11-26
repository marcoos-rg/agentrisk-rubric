"""AgentRisk rubric pipeline: an LLM answers a fixed questionnaire (rubric) about a proposed
agent action; a separate, learned model turns the answers into risk labels.

Modules
  taxonomy      official SemEval 2027 Task 10 label vocabularies (factors, drivers, domains)
  rubric        load + validate a rubric file from rubrics/
  prompt        system prompt, JSON schema, answer validation, feature encoding (hard and soft)
  logprobs      per-item answer probabilities from token logprobs
  data          download / load the AgentRisk splits
  run           run a rubric over traces against an OpenAI-compatible server (vLLM)
  analyze       summarise one run, or compare two runs
  check_rubric  validate a rubric and preview its prompt (no API calls)
"""
