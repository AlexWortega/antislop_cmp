"""Classify abnormal agent exits from real trial evidence; never invent a reward."""
import math

def confirmed_cap_grade(raw,session):
 if (raw.get('exception_info') or {}).get('exception_type')!='NonZeroAgentExitCodeError':return None
 messages=session.get('messages',[])
 reached=any(m.get('role')=='user' and str(m.get('content','')).startswith("You've reached the maximum number of tool-calling iterations allowed.") for m in messages)
 if not reached or not messages or messages[-1].get('role')!='assistant' or not str(messages[-1].get('content','')).strip():return None
 value=((raw.get('verifier_result') or {}).get('rewards') or {}).get('reward')
 if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:return None
 return {'valid':True,'reward':float(value),'source':'actual_terminal_verifier_after_confirmed_hermes_turn_cap','agent_turn_cap':True}

def _verifier_reward(raw):
 value=((raw.get('verifier_result') or {}).get('rewards') or {}).get('reward')
 if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:return None
 return float(value)

def confirmed_timeout_grade(raw):
 """An agent that outlives its time limit keeps the grade the real verifier gave it."""
 if (raw.get('exception_info') or {}).get('exception_type')!='AgentTimeoutError':return None
 value=_verifier_reward(raw)
 if value is None:return None
 return {'valid':True,'reward':value,'source':'actual_terminal_verifier_after_agent_timeout','agent_timeout':True}

def confirmed_rate_limit_label_grade(raw):
 """Harbor labels a run ApiRateLimitError from words in the agent's reply; the served model has no such limit, so the real verifier grade stands."""
 if (raw.get('exception_info') or {}).get('exception_type')!='ApiRateLimitError':return None
 value=_verifier_reward(raw)
 if value is None:return None
 return {'valid':True,'reward':value,'source':'actual_terminal_verifier_after_api_rate_limit_label','agent_rate_limit_label':True}

def docker_command_timed_out(raw):
 """True if a Docker call outlived Harbor's short limit before the verifier could run: host load, not the agent."""
 info=raw.get('exception_info') or {}
 return info.get('exception_type')=='RuntimeError' and str(info.get('exception_message','')).startswith('Command timed out after') and not raw.get('verifier_result')

def hermes_startup_failed(raw,session_text,agent_text):
 """True only if Hermes reported that it exited before sending any model message."""
 if (raw.get('exception_info') or {}).get('exception_type')!='NonZeroAgentExitCodeError' or session_text.strip():return False
 text=' '.join(agent_text.split())
 return text.startswith("Hermes couldn't start the model connection:") and 'Your message was not sent.' in text
