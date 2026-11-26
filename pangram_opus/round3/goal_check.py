"""Check the operational target saved before seeing fresh test labels."""
def check(report,criteria):
 checks={};m=report['overall']
 checks['overall_balanced_accuracy']=m['balanced_accuracy']>=criteria['overall_balanced_accuracy_min']
 checks['overall_agreement']=m['agreement']>=criteria['overall_agreement_min']
 checks['pangram_human_recall']=m['human_recall']>=criteria['each_teacher_class_recall_min']
 checks['pangram_ai_recall']=m['slop_recall']>=criteria['each_teacher_class_recall_min']
 human=report['by_origin'].get('human')
 checks['authentic_human_recall']=human is not None and human['human_recall'] is not None and human['human_recall']>=criteria['authentic_human_recall_min']
 for key,value in report['by_bucket'].items():
  positive=value['tp']+value['fn']+value['unavailable_positive'];negative=value['tn']+value['fp']+value['unavailable_negative']
  if min(positive,negative)>=criteria['length_min_cases_per_class']:checks['length:'+key]=value['balanced_accuracy']>=criteria['length_balanced_accuracy_min']
 for key,value in report['by_model'].items():
  if key!='not_applicable' and value['n']>=criteria['model_min_cases']:checks['model:'+key]=value['agreement']>=criteria['model_agreement_min']
 return {'met':all(checks.values()),'checks':checks,'failed':[k for k,v in checks.items() if not v],'criteria':criteria}
