class RewardModel:
    def calculate_reward(self, *args, **kwargs):
        raise NotImplementedError("Use get_reward instead.")

class RewardModelPPO(RewardModel):
    def __init__(self):

        self.REWARD_WIN = 50.0
        self.PENALTY_COMPROMISE = -50.0
        self.TIME_BONUS_FACTOR = 0.1
        self.PRECISION_WIN_ALPHA = 0.5

        self.PENALTY_INVALID = -3.0
        self.COST_FILTER = 0.5
        self.COST_PATCH = 1.5
        self.COST_RESTORE = 0.3
        self.PBRS_SCALE = 5.0
        self.PBRS_CLIP = 3.0
        self.RISK_TARGETING_BONUS = 2.0

        self.MAX_STEPS = 10

        self.total_reward = self.REWARD_WIN
        self.stack = 0

    def get_reward(self, env, action_type, success, outcome_type, target_is_critical,
                   ep_won, episode, node, **kwargs):
        """
        Three-component reward:
          1. Terminal win/loss (time-discounted)
          2. Availability cost (proportional to structural changes)
          3. PBRS on goal-node risk delta (clipped for stability)

        kwargs:
            step        : int   — current step (0-indexed)
            changes     : int   — structural changes from apply_action
            risk_before : float — goal-node risk before action
            risk_after  : float — goal-node risk after action
            gamma       : float — discount factor for PBRS
        """

        step = kwargs.get('step', 0)

        if outcome_type == 'compromise':
            return self.PENALTY_COMPROMISE * (1.0 + self.TIME_BONUS_FACTOR * (step + 1))

        if outcome_type == 'defense_success':
            steps_remaining = max(0, self.MAX_STEPS - 1 - step)
            ep_precision = kwargs.get('ep_precision', 0.0)
            precision_mult = self.PRECISION_WIN_ALPHA + (1.0 - self.PRECISION_WIN_ALPHA) * ep_precision
            return self.REWARD_WIN * (1.0 + self.TIME_BONUS_FACTOR * steps_remaining) * precision_mult

        if not success:
            return self.PENALTY_INVALID


        if action_type == 0:
            return 0.0

        reward = 0.0
        changes = kwargs.get('changes', 0)

        if action_type == 1:
            reward -= self.COST_FILTER * changes
        elif action_type == 2:
            reward -= self.COST_PATCH * changes
        elif action_type == 3:
            reward -= self.COST_RESTORE

        if action_type in (1, 2) and changes > 0:
            target_risk = kwargs.get('target_risk', 0.0)
            reward += self.RISK_TARGETING_BONUS * target_risk

        risk_before = kwargs.get('risk_before', None)
        risk_after = kwargs.get('risk_after', None)
        gamma = kwargs.get('gamma', 0.99)

        if risk_before is not None and risk_after is not None:
            pbrs = self.PBRS_SCALE * (risk_before - gamma * risk_after)
            pbrs = max(-self.PBRS_CLIP, min(self.PBRS_CLIP, pbrs))
            reward += pbrs

        return reward
