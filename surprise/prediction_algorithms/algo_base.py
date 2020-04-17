"""
The :mod:`surprise.prediction_algorithms.algo_base` module defines the base
class :class:`AlgoBase` from which every single prediction algorithm has to
inherit.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .. import similarities as sims
from .. import mySimilarities as mysims
from .predictions import PredictionImpossible
from .predictions import Prediction
from .optimize_baselines import baseline_als
from .optimize_baselines import baseline_sgd
import h5py

class AlgoBase(object):
    """Abstract class where is defined the basic behavior of a prediction
    algorithm.

    Keyword Args:
        baseline_options(dict, optional): If the algorithm needs to compute a
            baseline estimate, the ``baseline_options`` parameter is used to
            configure how they are computed. See
            :ref:`baseline_estimates_configuration` for usage.
    """

    def __init__(self, **kwargs):

        self.bsl_options = kwargs.get('bsl_options', {})
        self.sim_options = kwargs.get('sim_options', {})
        if 'user_based' not in self.sim_options:
            self.sim_options['user_based'] = True

    def fit(self, trainset):
        """Train an algorithm on a given training set.

        This method is called by every derived class as the first basic step
        for training an algorithm. It basically just initializes some internal
        structures and set the self.trainset attribute.

        Args:
            trainset(:obj:`Trainset <surprise.Trainset>`) : A training
                set, as returned by the :meth:`folds
                <surprise.dataset.Dataset.folds>` method.

        Returns:
            self
        """

        self.trainset = trainset

        # (re) Initialise baselines
        self.bu = self.bi = None

        return self

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        """Compute the rating prediction for given user and item.

        The ``predict`` method converts raw ids to inner ids and then calls the
        ``estimate`` method which is defined in every derived class. If the
        prediction is impossible (e.g. because the user and/or the item is
        unkown), the prediction is set according to :meth:`default_prediction()
        <surprise.prediction_algorithms.algo_base.AlgoBase.default_prediction>`.

        Args:
            uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
            iid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`.
            r_ui(float): The true rating :math:`r_{ui}`. Optional, default is
                ``None``.
            clip(bool): Whether to clip the estimation into the rating scale.
                For example, if :math:`\\hat{r}_{ui}` is :math:`5.5` while the
                rating scale is :math:`[1, 5]`, then :math:`\\hat{r}_{ui}` is
                set to :math:`5`. Same goes if :math:`\\hat{r}_{ui} < 1`.
                Default is ``True``.
            verbose(bool): Whether to print details of the prediction.  Default
                is False.

        Returns:
            A :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` object
            containing:

            - The (raw) user id ``uid``.
            - The (raw) item id ``iid``.
            - The true rating ``r_ui`` (:math:`\\hat{r}_{ui}`).
            - The estimated rating (:math:`\\hat{r}_{ui}`).
            - Some additional details about the prediction that might be useful
              for later analysis.
        """

        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:
            est = self.estimate(iuid, iiid)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def default_prediction(self):
        '''Used when the ``PredictionImpossible`` exception is raised during a
        call to :meth:`predict()
        <surprise.prediction_algorithms.algo_base.AlgoBase.predict>`. By
        default, return the global mean of all ratings (can be overridden in
        child classes).

        Returns:
            (float): The mean of all ratings in the trainset.
        '''

        return self.trainset.global_mean

    def test(self, testset, verbose=False):
        """Test the algorithm on given testset, i.e. estimate all the ratings
        in the given testset.

        Args:
            testset: A test set, as returned by a :ref:`cross-validation
                itertor<use_cross_validation_iterators>` or by the
                :meth:`build_testset() <surprise.Trainset.build_testset>`
                method.
            verbose(bool): Whether to print details for each predictions.
                Default is False.

        Returns:
            A list of :class:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` objects
            that contains all the estimated ratings.
        """

        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions

    def compute_baselines(self):
        """Compute users and items baselines.

        The way baselines are computed depends on the ``bsl_options`` parameter
        passed at the creation of the algorithm (see
        :ref:`baseline_estimates_configuration`).

        This method is only relevant for algorithms using :func:`Pearson
        baseline similarty<surprise.similarities.pearson_baseline>` or the
        :class:`BaselineOnly
        <surprise.prediction_algorithms.baseline_only.BaselineOnly>` algorithm.

        Returns:
            A tuple ``(bu, bi)``, which are users and items baselines."""

        # Firt of, if this method has already been called before on the same
        # trainset, then just return. Indeed, compute_baselines may be called
        # more than one time, for example when a similarity metric (e.g.
        # pearson_baseline) uses baseline estimates.
        if self.bu is not None:
            return self.bu, self.bi

        method = dict(als=baseline_als,
                      sgd=baseline_sgd)

        method_name = self.bsl_options.get('method', 'als')

        try:
            if getattr(self, 'verbose', False):
                print('Estimating biases using', method_name + '...')
            self.bu, self.bi = method[method_name](self)
            return self.bu, self.bi
        except KeyError:
            raise ValueError('Invalid method ' + method_name +
                             ' for baseline computation.' +
                             ' Available methods are als and sgd.')

    def compute_similarities(self):
        """Build the similarity matrix.

        The way the similarity matrix is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).

        This method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        Returns:
            The similarity matrix."""

        construction_func = {
            'cosine': sims.cosine,
            'msd': sims.msd,
            'pearson': sims.pearson,
            'pearson_baseline': sims.pearson_baseline
        }
        batched_construction_func = {
            'cosine': mysims.batched_cosine,
            'msd': mysims.batched_msd,
            'pearson': mysims.batched_pearson,
            'pearson_baseline': mysims.batched_pearson_baseline
        }
        if self.sim_options['user_based']:
            n_x, yr, n_y, xr = self.trainset.n_users, self.trainset.ir, self.trainset.n_items, self.trainset.ur
        else:
            n_x, yr, n_y, xr = self.trainset.n_items, self.trainset.ur, self.trainset.n_users, self.trainset.ir

        min_support = self.sim_options.get('min_support', 1)
        name = self.sim_options.get('name', 'msd').lower()

        if not self.sim_options.get('batched'):
            args = [n_x, yr, min_support]
            sim_func = construction_func[name]

        elif self.sim_options.get('batched'):
            try:
                file_path = self.sim_options['file_path']
            except KeyError:
                raise ValueError('file_path options required for batched calculations')
            batch_size = self.sim_options.get('batch_size', 1000)
            group_name = self.sim_options.get('group_name', 'similarity_matrix')
            dset_name = self.sim_options.get('dset_name', 'sims_'+name)
            if dset_name == 'sims_'+name:
                self.sim_options['dset_name'] = dset_name
            try:
                f = h5py.File(file_path, "w-")
            except OSError:
                print("File già esistente")
                raise ValueError('File {} già esistente'.format(file_path))
            try:
                sim_group = f.create_group(group_name)
            except ValueError:
                print("Gruppo 'similarity_matrix' già esistente")
                sim_group = f[group_name]
            sim_group.create_dataset(
                dset_name,
                shape=(n_x, n_x),
                dtype="f8", #floating point su 8byte -> np.double (float su 64 bit)
                chunks=(1, n_x), #1 chunk per ogni riga
                maxshape=(n_x, n_x), #Dimensione masssima nota a priori, numero utenti/oggetti x numero utenti/oggetti
                compression="gzip"
            )
            f.close()
            args = [n_x, yr, xr, min_support, batch_size, file_path, group_name, dset_name]
            sim_func = batched_construction_func[name]

        if name == 'pearson_baseline':
            shrinkage = self.sim_options.get('shrinkage', 100)
            bu, bi = self.compute_baselines()
            if self.sim_options['user_based']:
                bx, by = bu, bi
            else:
                bx, by = bi, bu
            args += [self.trainset.global_mean, bx, by, shrinkage]

        try:
            if getattr(self, 'verbose', False):
                print('Computing the {0} similarity matrix...'.format(name))
            sim = sim_func(*args)
            if getattr(self, 'verbose', False):
                print('Done computing similarity matrix.')
            return sim
        except KeyError:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')

    def get_neighbors(self, iid, k):
        """Return the ``k`` nearest neighbors of ``iid``, which is the inner id
        of a user or an item, depending on the ``user_based`` field of
        ``sim_options`` (see :ref:`similarity_measures_configuration`).

        As the similarities are computed on the basis of a similarity measure,
        this method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        For a usage example, see the :ref:`FAQ <get_k_nearest_neighbors>`.

        Args:
            iid(int): The (inner) id of the user (or item) for which we want
                the nearest neighbors. See :ref:`this note<raw_inner_note>`.

            k(int): The number of neighbors to retrieve.

        Returns:
            The list of the ``k`` (inner) ids of the closest users (or items)
            to ``iid``.
        """
        if self.sim_options['batched']:
            try:
                file_path = self.sim_options['file_path']
                group = self.sim_options.get('group_name', 'similarity_matrix')
                dset_name = self.sim_options['dset_name']
                f = h5py.File(file_path, "r")
            except OSError:
                raise ValueError("File {} inesistente".format(file_path))
            dset = f[group + '/' + dset_name]
            iid_distances = dset[iid, :]
            named_distances = [(i, d) for i,d in enumerate(iid_distances)]
            named_distances.sort(key=lambda tple: tple[1], reverse=True)
            k_nearest_neighbors = [i for (i, _) in named_distances[:k]]
            f.close()
        else:
            if self.sim_options['user_based']:
                all_instances = self.trainset.all_users
            else:
                all_instances = self.trainset.all_items

            others = [(x, self.sim[iid, x]) for x in all_instances() if x != iid]
            others.sort(key=lambda tple: tple[1], reverse=True)
            k_nearest_neighbors = [j for (j, _) in others[:k]]

        return k_nearest_neighbors
