"""
The :mod:`surprise.accuracy` module provides tools for computing accuracy
metrics on a set of predictions.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    rmse
    mse
    mae
    fcp
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import numpy as np
from six import iteritems


def rmse(predictions, verbose=True):
    """Compute RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mse(predictions, verbose=True):
    """Compute MSE (Mean Squared Error).

    .. math::
        \\text{MSE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse_ = np.mean([float((true_r - est)**2)
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MSE: {0:1.4f}'.format(mse_))

    return mse_


def mae(predictions, verbose=True):
    """Compute MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_

def _dcg(scores, discount=np.log2):
    """
    Compute the Discounted Cumulative Gain of a series of recommended items with rating scores.
    These should be relevance scores; they can be :math:`{0,1}` for binary relevance data.
    This is not a true top-N metric, but is a utility function for other metrics.
    Args:
        scores(array-like):
            The utility scores of a list of recommendations, in recommendation order.
        discount(ufunc):
            the rank discount function.  Each item's score will be divided the discount of its rank,
            if the discount is greater than 1.
    Returns:
        double: the DCG of the scored items.
    """
    scores = np.nan_to_num(scores)
    ranks = np.arange(1, len(scores) + 1)
    disc = discount(ranks)
    np.maximum(disc, 1, out=disc)
    np.reciprocal(disc, out=disc)
    return np.dot(scores, disc)

def ndcg(recs, truth, discount=np.log2, k=None):
    """
    Compute the normalized discounted cumulative gain :cite:p:`Jarvelin2002-xf`.
    Discounted cumultative gain is computed as:
    .. math::
        \\begin{align*}
        \\mathrm{DCG}(L,u) & = \\sum_{i=1}^{|L|} \\frac{r_{ui}}{d(i)}
        \\end{align*}
    Unrated items are assumed to have a utility of 0; if no rating values are provided in the
    truth frame, item ratings are assumed to be 1.
    This is then normalized as follows:
    .. math::
        \\begin{align*}
        \\mathrm{nDCG}(L, u) & = \\frac{\\mathrm{DCG}(L,u)}{\\mathrm{DCG}(L_{\\mathrm{ideal}}, u)}
        \\end{align*}
    This metric has a bulk implementation.
    Args:
        recs: The recommendation list.
        truth: The user's test data.
        discount(numpy.ufunc):
            The rank discount function.  Each item's score will be divided the discount of its rank,
            if the discount is greater than 1.
        k(int):
            The maximum list length.
            
    recs -> predictions
    
    """

    tpos = truth.index.get_indexer(recs['item'])

    if k is not None:
        recs = recs.iloc[:k]

    if 'rating' in truth.columns:
        i_rates = np.sort(truth.rating.values)[::-1]
        if k is not None:
            i_rates = i_rates[:k]
        ideal = _dcg(i_rates, discount)
        # make an array of ratings for this rec list
        r_rates = truth['rating'].values[tpos]
        r_rates[tpos < 0] = 0
        achieved = _dcg(r_rates, discount)
    else:
        ntrue = len(truth)
        if k is not None and ntrue > k:
            ntrue = k
        ideal = _fixed_dcg(ntrue, discount)
        tgood = tpos >= 0
        achieved = _dcg(tgood, discount)

    return achieved / ideal

def fcp(predictions, verbose=True):
    """Compute FCP (Fraction of Concordant Pairs).

    Computed as described in paper `Collaborative Filtering on Ordinal User
    Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
    and Sill, section 5.2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp))

    return fcp
