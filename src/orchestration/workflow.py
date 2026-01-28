# src/orchestration/workflow.py
from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING
from loguru import logger , logging
from langgraph.graph import StateGraph, END

if TYPE_CHECKING:
    from core.intent_classifier import IntentClassifier
    from core.retriever import HybridRetriever
    from core.reranker import LegalReranker
    from core.llm_handler import LegalLLMHandler
    from validation.answer_validator import AnswerValidator

class LegalRAGWorkflow:
    """LangGraph-based orchestration workflow."""
    
    def __init__(self, 
                 intent_classifier: IntentClassifier,
                 retriever: HybridRetriever,
                 reranker: LegalReranker,
                 llm_handler: LegalLLMHandler,
                 validator: AnswerValidator):
        
        self.intent_classifier = intent_classifier
        self.retriever = retriever
        self.reranker = reranker
        self.llm_handler = llm_handler
        self.validator = validator
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow."""
        
        workflow = StateGraph(dict)

        
        # Define nodes
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate", self._validate_node)
        
        # Define edges
        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_edge("validate", END)
        
        return workflow.compile()
    
    def _classify_intent_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Intent classification node."""
        try:
            logger.info(f"Classifying query: {state['query']}")
            intent = self.intent_classifier.classify(state['query'])
            state['intent'] = intent.dict()
            logger.info(f"Detected domain: {intent.domain}, law: {intent.law_type}")
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            state['error'] = str(e)
        return state
    
    def _retrieve_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hybrid retrieval node."""
        try:
            logger.info("Retrieving relevant chunks")
            candidates = self.retriever.retrieve(state['query'], top_k=15)
            state['candidates'] = candidates
            logger.info(f"Retrieved {len(candidates)} candidates")
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            state['error'] = str(e)
        return state
    
    def _rerank_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Reranking node."""
        try:
            logger.info("Reranking candidates")
            reranked = self.reranker.rerank(
                state['query'], 
                state['candidates'], 
                top_k=5
            )
            state['final_chunks'] = reranked
            logger.info(f"Reranked to {len(reranked)} chunks")
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            state['error'] = str(e)
        return state
    
    def _generate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """LLM generation node."""
        try:
            logger.info("Generating answer")
            context = self.llm_handler.build_context(state['final_chunks'])
            answer = self.llm_handler.generate_answer(state['query'], context)
            state['answer'] = answer
            state['context'] = context
            logger.info("Answer generated")
        except Exception as e:
            logger.error(f"Generation error: {e}")
            state['error'] = str(e)
        return state
    
    def _validate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validation node."""
        try:
            logger.info("Validating answer")
            validation = self.validator.validate(
                state['answer'], 
                state['final_chunks']
            )
            state['validation'] = validation
            logger.info(f"Validation: {validation['valid']}, Confidence: {validation['confidence']}")
        except Exception as e:
            logger.error(f"Validation error: {e}")
            state['error'] = str(e)
        return state
    
    def run(self, query: str) -> Dict[str, Any]:
        """Execute workflow."""
        initial_state = {"query": query}
        final_state = self.graph.invoke(initial_state)
        return final_state