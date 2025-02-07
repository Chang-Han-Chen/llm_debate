import logging
from typing import List, Tuple, Optional

from core.agents.debater_quality import DebaterQuality
from core.agents.judge_base import JudgeBase
from core.rollouts.utils import CacheManager, TranscriptConfig

LOGGER = logging.getLogger(__name__)

class DebaterTreeSearch(DebaterQuality):
    """Implements tree search for argument selection instead of flat BoN."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract tree search config with better defaults and validation
        tree_search_config = getattr(self.config, "tree_search", None)
        if tree_search_config:
            self.root_branching = tree_search_config.root_branching  # New: separate root branching
            self.child_branching = tree_search_config.child_branching  # New: child node branching
            self.depth = tree_search_config.depth
            self.exploration_temp = tree_search_config.exploration_temp
        else:
            LOGGER.warning("No tree_search config found, using defaults")
            self.root_branching = 4  # Matches BoN experiment default
            self.child_branching = 2
            self.depth = 2
            self.exploration_temp = 0.8

    async def _expand_node(self, node: DebateNode) -> List[DebateNode]:
        """Generate child nodes for the given node."""
        debate_history = node.get_debate_history()
        branching = self.root_branching if not node.parent else self.child_branching
        
        # Format the prompt based on whether this is opponent's turn or not
        if node.is_opponent:
            prompt = self.config.tree_search.opponent_prompt.format(
                previous_argument=node.parent.response,
                debate_history="\n".join(debate_history[:-1])  # Exclude current node
            )
        else:
            prompt = self.config.tree_search.response_prompt.format(
                opponent_argument=node.parent.response if node.parent else None,
                debate_history="\n".join(debate_history)
            )
        
        # Generate responses using the LLM
        responses = await self.get_completion_with_retries(
            prompt, 
            n=branching,
            temperature=self.config.tree_search.exploration_temp
        )
        
        # Create child nodes
        children = []
        for response in responses:
            child = node.add_child(response, is_opponent=not node.is_opponent)
            children.append(child)
        
        return children

    async def get_completion_with_retries(self, prompt: str, n: int, temperature: float) -> List[str]:
        """Generate responses with retries using the configured LLM."""
        # Use existing API handler for consistency with BoN implementation
        responses = await self.api_handler(
            model_ids=self.config.language_model.model,
            prompt=prompt,
            temperature=temperature,
            top_p=self.config.language_model.top_p,
            max_tokens=self.config.language_model.max_tokens,
            n=n,
            num_candidates_per_completion=self.config.language_model.num_candidates_per_completion,
            is_valid=self.is_valid,
            insufficient_valids_behaviour="pad_invalids",
        )
        return [x.completion.strip() for x in responses]

    async def _evaluate_node(
        self,
        response: str,
        transcript: TranscriptConfig,
        judge: JudgeBase,
    ) -> float:
        """Get score for a node using the judge."""
        argument = self.extract_argument(response)
        truncated_argument = self.truncate(argument)
        return await judge.get_argument_rating(
            transcript,
            truncated_argument, 
            self.side,
            self.method,
        )

    async def _tree_search(
        self,
        initial_responses: List[str],
        transcript: TranscriptConfig,
        judge: JudgeBase,
    ) -> Tuple[str, str, dict]:  # Added dict for metrics
        """Implements tree search with configurable depth and branching factors."""
        search_trace = []
        metrics = {
            "total_nodes": 0,
            "depth_reached": 0,
            "avg_scores": [],
            "best_path_scores": [],
        }
        
        # Level 0: Create and evaluate root nodes
        root_nodes = [DebateNode(resp, is_opponent=False) for resp in initial_responses]
        metrics["total_nodes"] += len(root_nodes)
        
        for node in root_nodes:
            node.score = await self._evaluate_node(node.response, transcript, judge)
            search_trace.append(f"Root node: score={node.score:.3f}")
        metrics["avg_scores"].append(sum(n.score for n in root_nodes) / len(root_nodes))
        
        # Expand tree to configured depth
        current_level = root_nodes
        for depth in range(1, self.depth + 1):
            metrics["depth_reached"] = depth
            next_level = []
            branching = self.child_branching
            
            for parent in current_level:
                children = await self._expand_node(parent)
                metrics["total_nodes"] += len(children)
                
                for child in children:
                    child.score = await self._evaluate_node(
                        child.response,
                        self.create_modified_transcript(transcript, child.get_debate_history()),
                        judge
                    )
                    next_level.append(child)
                    
                    trace_prefix = "Opponent" if child.is_opponent else "Our"
                    search_trace.append(
                        f"{trace_prefix} response at depth {depth}: score={child.score:.3f}"
                    )
            
            if next_level:
                metrics["avg_scores"].append(
                    sum(n.score for n in next_level) / len(next_level)
                )
            current_level = next_level

        # Find best leaf node and track scores along its path
        leaf_nodes = [n for n in current_level if not n.children]
        best_leaf = max(leaf_nodes, key=lambda n: n.score)
        
        # Track scores along best path
        node = best_leaf
        while node:
            metrics["best_path_scores"].insert(0, node.score)
            node = node.parent

        debate_history = best_leaf.get_debate_history()
        search_trace = "\n".join(search_trace)
        
        return debate_history[0], search_trace, metrics

    async def take_turn(
        self,
        transcript: TranscriptConfig,
        current_step: int,
        cache_manager: CacheManager,
        judge: JudgeBase = None,
        judge_critic: JudgeBase = None,
        judge_critique_pm: JudgeBase = None,
    ):
        """Override parent method to use tree search."""
        # Get initial responses using parent BoN mechanism
        responses = None
        response_key = f"responses_{self.side}"
        if current_step < len(cache_manager.results):
            responses = cache_manager.results[current_step].get(response_key, None)
        if responses is None:
            responses = await self.get_completion(transcript)
            cache_manager.save_item(current_step, response_key, responses)

        assert len(responses) == self.config.BoN, f"Got {len(responses)} responses, expected {self.config.BoN}"

        # Run tree search
        best_response, search_trace, metrics = await self._tree_search(responses, transcript, judge)
        argument = self.extract_argument(best_response)
        truncated_argument = self.truncate(argument)

        # Handle critique refinement if enabled
        if self.config.cBoN > 0:
            critique = await self.get_critique(
                truncated_argument,
                transcript,
                current_step,
                cache_manager,
                judge_critic,
                judge_critique_pm,
            )
            refinements = None
            refinement_key = f"refinement_{self.side}"
            if current_step < len(cache_manager.results):
                refinements = cache_manager.results[current_step].get(refinement_key, None)
            if refinements is None:
                refinements = await self.get_refinements(transcript, best_response, critique)
                cache_manager.save_item(current_step, refinement_key, refinements)

            if self.config.BoN > 1:
                refinement, refinements_string = await self.judge_preference(
                    refinements, transcript, judge, strict=False
                )
            else:
                refinement = refinements[0]
                refinements_string = refinement

            if "<argument>" in refinement:
                argument = self.extract_argument(refinement)
                truncated_argument = self.truncate(argument)
            else:
                LOGGER.warning(f"Refinement had issue. Using original argument instead.")

            search_trace += f"\n\nCritique:\n{critique}\n\nRefinements:\n{refinements_string}"

        return truncated_argument, search_trace

    def create_modified_transcript(
        self, 
        transcript: TranscriptConfig,
        response: str
    ) -> TranscriptConfig:
        """Helper to create a transcript copy with the given response."""
        modified_transcript = transcript.copy(deep=True)
        argument = self.extract_argument(response)
        if self.correct:
            modified_transcript.rounds[-1].correct = argument
        else:
            modified_transcript.rounds[-1].incorrect = argument
        return modified_transcript 

class DebateNode:
    def __init__(self, response: str, parent: Optional['DebateNode'] = None, is_opponent: bool = False):
        self.response = response
        self.score = None
        self.parent = parent
        self.children = []
        self.is_opponent = is_opponent
    
    def get_debate_history(self) -> List[str]:
        """Walks up the tree to construct full debate history."""
        history = []
        current = self
        while current:
            history.append(current.response)
            current = current.parent
        return list(reversed(history))

    def add_child(self, response: str, is_opponent: bool) -> 'DebateNode':
        """Creates and adds a child node."""
        child = DebateNode(response, parent=self, is_opponent=is_opponent)
        self.children.append(child)
        return child 