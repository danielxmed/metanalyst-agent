"""State reducers for managing complex state updates in Metanalyst-Agent"""

from typing import List, Dict, Any, Union
from langchain_core.messages import BaseMessage


def add_messages(existing: List[BaseMessage], new: List[BaseMessage]) -> List[BaseMessage]:
    """
    Reducer for adding new messages to existing message list
    
    Args:
        existing: Current list of messages
        new: New messages to add
        
    Returns:
        Combined list of messages
    """
    if not existing:
        return new or []
    if not new:
        return existing
    return existing + new


def merge_dicts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reducer for merging dictionaries with deep merge for nested dicts
    
    Args:
        existing: Current dictionary
        new: New dictionary to merge
        
    Returns:
        Merged dictionary
    """
    if not existing:
        return new or {}
    if not new:
        return existing
    
    result = existing.copy()
    
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def append_list(existing: List[Any], new: Union[List[Any], Any]) -> List[Any]:
    """
    Reducer for appending items to a list
    
    Args:
        existing: Current list
        new: New item(s) to append
        
    Returns:
        Extended list
    """
    if not existing:
        existing = []
    
    if isinstance(new, list):
        return existing + new
    elif new is not None:
        return existing + [new]
    else:
        return existing


def increment_counter(existing: Dict[str, int], new: Dict[str, int]) -> Dict[str, int]:
    """
    Reducer for incrementing counters in a dictionary
    
    Args:
        existing: Current counter dictionary
        new: Increments to apply
        
    Returns:
        Updated counter dictionary
    """
    if not existing:
        existing = {}
    
    result = existing.copy()
    
    for key, increment in (new or {}).items():
        result[key] = result.get(key, 0) + increment
    
    return result


def update_metrics(existing: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    """
    Reducer for updating metrics with validation
    
    Args:
        existing: Current metrics
        new: New metric values
        
    Returns:
        Updated metrics dictionary
    """
    if not existing:
        existing = {}
    
    result = existing.copy()
    
    for key, value in (new or {}).items():
        if isinstance(value, (int, float)):
            result[key] = float(value)
    
    return result