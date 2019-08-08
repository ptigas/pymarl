from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .basic_controller import BasicMAC

# This multi-agent controller with communication shares parameters between agents
class ComMAC(BasicMAC):
    def communication_pool(self, pool='mean'):
        hidden_states = self.hidden_states.view(self.args.batch_size, self.n_agents, -1)

        hs = hidden_states.shape
        if pool == 'mean':
            message = th.mean(hidden_states, dim=1)
        else:
            message = th.zeros((hs[0], hs[1], hs[2]))

        return message

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # Append communication to input
        if self.args.communication_pool:
            communication = self.communication_pool()
            if t == 0:
                # for t=0, zero the communication. the hidden state is init to 0 already
                # but this allows for extra control
                inputs.append(th.zeros_like(communication).unsqueeze(1).expand(-1, self.n_agents, -1))
            else:
                inputs.append(communication.unsqueeze(1).expand(-1, self.n_agents, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]

        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.communication_pool:
            input_shape += self.args.rnn_hidden_dim

        return input_shape
