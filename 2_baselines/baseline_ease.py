"""Linear item-item autoencoder with zero diagonal (EASE), closed form on history before test."""
import candidates
import metrics
import stage

start, history, scored_users, pool_items, frame = stage.prepare("test")

interactions = candidates.matrix(history, stage.SHAPE)
weights = candidates.ease(interactions)

frame["score"] = candidates.scores(scored_users, pool_items, interactions, weights)

metrics.report("ease", frame, "score")
