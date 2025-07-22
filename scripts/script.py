class winningAdSelector:
    def __init__(self, df):
        self.df = df

    def get_winning_ads(self, top_n=10):
        winning = self.df[
            (self.df['amount_spent'] > 100) &
            (self.df['max_roas'] >= 2) &
            (self.df['result_rate_(%)'] > 20) &
            (self.df['scroll_stop_(%)'] > 25) &
            (self.df['hook_rate_(%)'] > 30) &
            (self.df['hold_rate_(%)'] > 10) &
            (self.df['cpr'] <= 20) &
            (self.df['ctr'] > 2) &
            (self.df['cpc'] < 1.5)
        ].copy()
        winning_sorted = winning.sort_values(by='result_rate_(%)', ascending=False).head(top_n)
        selected_cols = ['ad_name', 'ad_set_name', 'amount_spent', 'max_roas', 'result_rate_(%)',
                         'scroll_stop_(%)', 'hook_rate_(%)', 'hold_rate_(%)', 'ctr', 'cpr', 'cpc']
        return winning_sorted[selected_cols].round(2)  # Keep numbers clean


def highlight_column(s, column_to_highlight):
    return ['background-color: #F0F2F6' if col == column_to_highlight else '' for col in s.index]


