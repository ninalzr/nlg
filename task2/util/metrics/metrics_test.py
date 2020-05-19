from task2.util.validation_metrics import evaluate

y_gold =
score, eval = evaluate(y_gold, y_predicted, model.tgt_lookup, cut_at_eos=True, use_accuracy=False, use_bleu=True)