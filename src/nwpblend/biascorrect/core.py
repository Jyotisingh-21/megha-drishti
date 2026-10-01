import logging
import pickle

import xarray as xr

from nwpblend.biascorrect.lapse_rate import correct_t2m
from nwpblend.biascorrect.quantile_map import QuantileMapper

logger = logging.getLogger(__name__)


class BiasCorrector:
    def __init__(self, lapse_rate=0.0065):
        self.lapse_rate = lapse_rate
        self.qm = QuantileMapper()

    def fit(
        self,
        ds_fcst: xr.Dataset,
        ds_truth: xr.Dataset,
        oro_model: xr.DataArray = None,
        oro_truth: xr.DataArray = None,
    ):
        """
        Fit the bias corrector.
        """
        # First, apply deterministic lapse rate correction to the forecast
        if oro_model is not None and oro_truth is not None:
            ds_fcst = correct_t2m(ds_fcst, oro_model, oro_truth, self.lapse_rate)

        # Then fit the quantile mapper on the residuals
        logger.info("Fitting QuantileMapper...")
        self.qm.fit(ds_fcst, ds_truth)
        return self

    def transform(
        self, ds_fcst: xr.Dataset, oro_model: xr.DataArray = None, oro_truth: xr.DataArray = None
    ) -> xr.Dataset:
        """
        Apply the bias corrector.
        """
        if oro_model is not None and oro_truth is not None:
            ds_fcst = correct_t2m(ds_fcst, oro_model, oro_truth, self.lapse_rate)

        logger.info("Transforming with QuantileMapper...")
        return self.qm.transform(ds_fcst)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
