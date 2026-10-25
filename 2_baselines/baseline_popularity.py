"""Lower bound: every user gets the same scores, the number of distinct viewers before the window."""
import candidates
import metrics
import stage

start, history, scored_users, pool_items, frame = stage.prepare("test")

scored = frame.merge(
    candidates.popular(history).rename("score"), on="item", how="left"
)

metrics.report("popularity", scored, "score")
