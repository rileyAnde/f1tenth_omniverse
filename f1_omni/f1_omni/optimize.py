import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import rclpy
from rl_train import dq
import joblib

def optimize_sac(trial):
    
    # define hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_uniform("gamma", 0.95, 0.999)
    tau = trial.suggest_uniform("tau", 0.005, 0.05)
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
    net_arch = trial.suggest_categorical("net_arch", [[256, 256], [400, 300], [256, 256, 128]])

    # initialize ROS2 and the environment
    #rclpy.init()
    env = dq()

    # create the SAC model with the sampled hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        buffer_size=buffer_size,
        policy_kwargs={"net_arch": net_arch},
        verbose=0,
    )

    # train the model for a fixed number of steps
    model.learn(total_timesteps=50000)

    # Evaluate the model performance (higher return is better)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    
    # Clean up
    env.destroy_node()
    #rclpy.shutdown()
    return mean_reward  # Optuna tries to maximize this value


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_sac, n_trials=20)

    joblib.dump(study, "optuna_sac_study.pkl")

    print("Best Hyperparameters:", study.best_params)