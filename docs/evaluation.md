# Evaluation protocol

For prediction `(xp,yp)` and ground truth `(xt,yt)`:

`error = sqrt((xp-xt)^2 + (yp-yt)^2)`

Accuracy at a chosen tolerance:

`100 * successful_predictions / total_predictions`

Report test count, tolerance, accuracy, mean/median error, and inference time for a 1000x1000 Search image.

Include one honest success and one honest failure in the presentation. Never fabricate benchmark values.
