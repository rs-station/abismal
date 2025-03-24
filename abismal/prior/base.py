import tf_keras as tfk


class PriorBase(tfk.layers.Layer):
    def distribution(self, asu_id, hkl):
        raise NotImplementedError(
            "Derived classes must implement distribution(asu_id, hkl) -> Distribution")

    def flat_distribution(self):
        raise NotImplementedError(
            "Derived classes must implement flat_distribution() -> Distribution")
