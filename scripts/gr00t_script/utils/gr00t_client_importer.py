"""
gr00t_client_importer.py

A compatibility wrapper for connecting IsaacLab to different GR00T policy server versions (e.g. N1.5, N1.6).

This class hides API and import differences between GR00T versions and exposes
a unified `get_action(obs)` interface to the caller.
"""

class Gr00tClient:
    """
    Gr00tClient is a compatibility wrapper for GR00T policy servers.

    It allows IsaacLab to connect to different GR00T versions (e.g. N1.5, N1.6) using a unified interface, hiding version-specific import paths and API differences.

    Parameters
    ----------
    version : str
        GR00T version identifier (e.g. "N1.5", "N1.6").
    host : str
        Policy server host address (default: "localhost").
    port : int
        Policy server port (default: 5555).
    """

    def __init__(self, version: str, host="localhost", port=5555):
        self.version = version.lower()

        try:
            if self.version in ("n1.5", "1.5", "v1.5"):
                from gr00t.eval.robot import RobotInferenceClient
                self._client = RobotInferenceClient(host=host, port=port)
                self._mode = "N1.5"
            elif self.version in ("n1.6", "1.6", "v1.6"):
                from gr00t.policy.server_client import PolicyClient
                self._client = PolicyClient(host=host, port=port)
                self._mode = "N1.6"
            else:
                raise ValueError(f"Unsupported gr00t version: {version}")
            
        except ModuleNotFoundError as e:
            raise ImportError(
                f"\n[GR00T VERSION MISMATCH]\n"
                f"You requested GR00T {self.version} client, but the client implementation "
                f"cannot be found in the current GR00T environment.\n"
                f"Please check that:\n"
                f"  - Your gr00t server environment is checked out to the {self.version} branch\n"
                f"  - OR your Python environment has GR00T {self.version} installed\n"
            ) from e
        
        print(f"\nTrying to connect to the GR00T {self._mode} policy server.")
        if not self._client.ping():
            raise RuntimeError("Cannot connect to the gr00t policy server")
        else:
            print(f"\nSuccessfully connect to the GR00T {self._mode} policy server.")

        modality_configs = self._client.get_modality_config()
        print(f"Retrieved modality keys: {list(modality_configs.keys())}")
        
    def get_action(self, obs):
        if self._mode == "N1.5":
            action = self._client.get_action(obs)
            return action

        elif self._mode == "N1.6":
            action, info = self._client.get_action(obs)
            return action
