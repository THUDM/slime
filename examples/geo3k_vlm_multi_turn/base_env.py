class BaseInteractionEnv:
    """
    Base class that defines the explicit contract for interaction environments.
    """

    def reset(self):
        raise NotImplementedError

    def step(self, response_text: str):
        raise NotImplementedError

    def close(self):
        pass

    def format_observation(self, observation: dict) -> dict:
        observation = observation or {}
        content = []
        multimodal = observation.get("multi_modal_data") or {}

        for _, images in multimodal.items():
            for image in images:
                content.append({"type": "image", "image": image})

        content.append({"type": "text", "text": observation.get("obs_str", "")})
        return {"role": "user", "content": content}
