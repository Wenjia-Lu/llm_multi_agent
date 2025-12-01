# Confidence-Gated Multi-Agent Debate System - Implementation Plan

## Overview
This document outlines the implementation plan for a confidence-gated multi-agent debate system that evaluates mathematical reasoning tasks. The system starts with Llama 3.1 8B as both the gate model and debate agent, using a 0.95 confidence threshold to trigger debates.

## Key Requirements

### Core Functionality
- **Confidence-Gated Initiation**: Llama 3.1 8B evaluates initial confidence before triggering multi-agent debate
- **Fixed Multi-Agent Debate**: 3-agent debate system with predetermined agents
- **Confidence Tracking**: Each agent appends confidence scores to their responses
- **Caching**: All agents are loaded once and cached for reuse across questions
- **Configurable Parameters**: Command-line arguments for gate model, confidence threshold, and debate rounds

### Target Models (Initial Implementation)
- **Gate Model**: llama3.1-8b-instruct (for confidence evaluation)
- **Debate Agents** (Fixed Set):
  - llama3.1-8b-instruct (general purpose)
  - DeepSeek-R1-Qwen-7B (specialized reasoning)
  - Mathstral-7B (mathematical reasoning)

### Default Parameters
- **Confidence Threshold**: 0.95
- **Debate Rounds**: 3
- **Gate Model**: llama3.1-8b-instruct

## Implementation Stages

### Stage 1: Core Architecture and Agent Caching ✅ COMPLETED
**Objective**: Build the basic debate system with fixed 3-agent setup and caching

**Requirements**:
- [x] Modified existing `gen_OS.py` to use fixed 3-agent array:
  - llama3.1-8b-instruct (gate + debate agent)
  - DeepSeek-R1-Qwen-7B
  - Mathstral-7B
- [x] Added command-line arguments for:
  - `--confidence_threshold` (default: 0.95)
  - `--debate_rounds` (default: 3)
- [x] Implemented agent loading and caching in array
- [x] Created config files for missing models

**Deliverables**:
- Modified `extension/gsm/gen_OS.py` with agent array
- Agent config files: `DeepSeek-R1-Qwen-7B.json`, `Mathstral-7B.json`
- Command-line interface with configurable parameters
- Basic debate system structure

**End State**: System can load and cache 3 agents, ready for confidence gating

---

### Stage 2: Confidence Gating Implementation
**Objective**: Implement confidence evaluation and gating logic for Llama 3.1 8B

**Requirements**:
- [ ] Create confidence evaluation prompts for GSM8K math problems
- [ ] Implement confidence score extraction from Llama 3.1 8B responses
- [ ] Add confidence threshold comparison (default 0.95)
- [ ] Handle confidence parsing for different response formats
- [ ] Add fallback logic for missing or malformed confidence scores

**Deliverables**:
- Confidence evaluation function
- Gate logic that decides when to debate
- Logging for confidence decisions

**End State**: System can evaluate confidence and decide whether to proceed with debate

---

### Stage 3: Multi-Agent Debate Engine
**Objective**: Build the 3-agent debate coordination system

**Requirements**:
- [ ] Implement fixed 3-agent debate rounds (configurable number)
- [ ] Create debate prompt templates that share previous agent responses
- [ ] Add confidence score appending to each agent's response
- [ ] Implement debate flow: each agent sees previous agents' answers
- [ ] Handle debate termination after specified rounds

**Deliverables**:
- Multi-round debate coordination
- Agent response sharing mechanism
- Confidence score integration in responses

**End State**: Functional 3-agent debate system with confidence tracking

---

### Stage 4: GSM8K Integration
**Objective**: Integrate with existing GSM8K evaluation pipeline

**Requirements**:
- [ ] Adapt existing `gen_OS.py` structure to use confidence debate system
- [ ] Create debate-aware result storage format
- [ ] Maintain compatibility with existing evaluation scripts
- [ ] Add confidence and debate metadata to results

**Deliverables**:
- Modified GSM8K generation script
- Updated result format with confidence scores
- Backward compatibility maintained

**End State**: System can process GSM8K dataset with confidence gating and debate

---

### Stage 5: Testing and Validation
**Objective**: Test the system and validate against single-agent baseline

**Requirements**:
- [ ] Test confidence gating with different thresholds
- [ ] Validate debate improves accuracy over single-agent
- [ ] Performance testing (memory usage, latency)
- [ ] Error handling and edge case testing

**Deliverables**:
- Test results comparing debate vs single-agent accuracy
- Performance benchmarks
- Confidence threshold analysis

**End State**: Validated system ready for experimentation with different parameters

## Command-Line Interface

The system will support the following command-line arguments:

```bash
python confidence_debate.py \
  --gate_model llama3.1-8B-instruct \
  --confidence_threshold 0.95 \
  --debate_rounds 3 \
  --config_base_path llm/configs/
```

### Configuration Schema

#### Debate Configuration (JSON for future extension)
```json
{
  "gate_model": "llama3.1-8B-instruct",
  "debate_agents": [
    "llama3.1-8B-instruct",
    "DeepSeek-R1-Qwen-7B",
    "Mathstral-7B"
  ],
  "confidence_threshold": 0.95,
  "max_debate_rounds": 3,
  "task_type": "gsm8k"
}
```

#### Required Model Config Files
- `llm/configs/llama3.1-8B-instruct.json`
- `llm/configs/DeepSeek-R1-Qwen-7B.json` (needs to be created)
- `llm/configs/Mathstral-7B.json` (needs to be created)

## Success Criteria

### Functional Requirements
- [ ] System loads and caches 3 fixed agents without reloading between questions
- [ ] Llama 3.1 8B can evaluate confidence with 0.95 threshold gating
- [ ] 3-agent debate completes successfully with configurable rounds
- [ ] All responses include confidence scores appended to answers
- [ ] Command-line arguments work for gate model, threshold, and rounds

### Performance Requirements
- [ ] Agent caching eliminates model reloading overhead
- [ ] System processes GSM8K questions without excessive memory usage
- [ ] Debate rounds complete within reasonable time per question

### Quality Requirements
- [ ] Confidence gating works reliably for GSM8K problems
- [ ] Debate responses are properly formatted with confidence scores
- [ ] System handles edge cases (parsing failures, model errors)
- [ ] Results are reproducible with same parameters

## Risk Assessment

### Technical Risks
- **Memory Usage**: 3 large models may exceed available RAM
- **Missing Models**: DeepSeek-R1-Qwen-7B and Mathstral-7B configs don't exist yet
- **Confidence Parsing**: Llama 3.1 8B may not provide reliable confidence scores

### Mitigation Strategies
- Start with CPU-only configurations and quantization
- Create placeholder configs and test with available models first
- Implement fallback confidence estimation if parsing fails

## Timeline Estimate
- Stage 1: 3-4 days (Agent caching and CLI)
- Stage 2: 2-3 days (Confidence gating)
- Stage 3: 3-4 days (Debate engine)
- Stage 4: 2-3 days (GSM8K integration)
- Stage 5: 2-3 days (Testing and validation)

**Total**: 2-3 weeks

## Dependencies
- Existing LLM framework (`llm/` directory)
- Llama 3.1 8B config file (exists)
- Need to create DeepSeek-R1-Qwen-7B and Mathstral-7B config files
- CPU with sufficient RAM for 3 models (or implement offloading)
