def plot_soc_across_episodes(self, battery_data, save_path=None):
    """Plot battery SOC profiles for selected episodes"""
    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    try:
        # Convert list of DataFrames to a single DataFrame
        all_episodes_df = pd.concat(battery_data, ignore_index=True)
        
        # Verify columns
        logger.info(f"Available columns: {all_episodes_df.columns.tolist()}")
        
        if 'Episode' not in all_episodes_df.columns:
            logger.error("Episode column not found in DataFrame")
            return None

        # Get unique episode numbers and select every 10th episode
        episodes = sorted(all_episodes_df['Episode'].unique())
        selected_episodes = episodes[::10]
        
        if len(selected_episodes) == 0:
            logger.error("No episodes found in data")
            return None

        # Create colormap for episodes
        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_episodes)))
        
        final_socs = []
        
        # Plot SOC for selected episodes
        for idx, episode in enumerate(selected_episodes):
            episode_data = all_episodes_df[all_episodes_df['Episode'] == episode]
            if len(episode_data) > 0:
                plt.plot(episode_data['Time_Hour'], 
                        episode_data['Battery_SOC'] * 100,
                        color=colors[idx],
                        linewidth=2,
                        label=f'Episode {episode}')
                final_socs.append(episode_data.iloc[-1]['Battery_SOC'] * 100)

        # Add limit lines
        plt.axhline(y=self.SOC_MAX * 100, color='red', linestyle='--', alpha=0.5,
                  label='Maximum SOC')
        plt.axhline(y=self.SOC_MIN * 100, color='red', linestyle='--', alpha=0.5,
                  label='Minimum SOC')
        plt.axhline(y=self.SOC_OPTIMAL_MAX * 100, color='green', linestyle=':', alpha=0.5,
                  label='Optimal Max')
        plt.axhline(y=self.SOC_OPTIMAL_MIN * 100, color='green', linestyle=':', alpha=0.5,
                  label='Optimal Min')

        # Calculate metrics for final episode
        if len(selected_episodes) > 0:
            final_episode_data = all_episodes_df[all_episodes_df['Episode'] == max(selected_episodes)]
            time_in_optimal = ((final_episode_data['Battery_SOC'] >= self.SOC_OPTIMAL_MIN) & 
                             (final_episode_data['Battery_SOC'] <= self.SOC_OPTIMAL_MAX)).mean() * 100
            avg_soc = final_episode_data['Battery_SOC'].mean() * 100
            soc_variance = final_episode_data['Battery_SOC'].var() * 100

            metrics_text = (
                f"Final Episode Metrics:\n"
                f"Average SOC: {avg_soc:.1f}%\n"
                f"SOC Variance: {soc_variance:.2f}%\n"
                f"Time in Optimal Range: {time_in_optimal:.1f}%"
            )

            plt.text(1.02, 0.5, metrics_text,
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                    verticalalignment='center')

        plt.xlabel('Time (Hour)')
        plt.ylabel('State of Charge (%)')
        plt.title('Battery SOC Profiles Across Episodes')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.ylim(self.SOC_MIN * 95, self.SOC_MAX * 105)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt.gcf()

    except Exception as e:
        logger.error(f"Error in plot_soc_across_episodes: {str(e)}")
        return None