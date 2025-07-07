import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

class ReturnAnalytic(pd.Series):
    def __init__(self, ret, name=None):

        if not isinstance(ret, pd.Series):
            raise ValueError("return_series must be a pandas Series")
        
        if ret.index.dtype != 'datetime64[ns]':
            raise ValueError("index must be datetime64[ns]")
        super().__init__(ret)
        
        ret = ret.fillna(0)
        self.ret = ret
        self.name = name if name is not None else 'Strategy'
        self.start_date = ret.index[0]
        self.end_date = ret.index[-1]
        self.stats = Stats(self)
        self.plots = Plots(self)
    
    def to_cumulative_value(self):
        """Convert returns to cumulative value (growth of $1 invested)"""
        return (1 + self.ret).cumprod()
    
    def to_drawdown(self):
        """Calculate drawdown series"""
        cumulative = self.to_cumulative_value()
        return (cumulative / cumulative.expanding().max()) - 1

    def to_return(self, freq='D'):
        """Convert returns to specified frequency"""
        if freq == 'D':
            return self.ret
        elif freq == 'M':
            # Monthly returns - group by month and compound
            monthly_ret = self.ret.groupby(self.ret.index.to_period('M')).apply(
                lambda x: (1 + x).prod() - 1
            )
            return monthly_ret
        elif freq == 'Q':
            # Quarterly returns - group by quarter and compound
            quarterly_ret = self.ret.groupby(self.ret.index.to_period('Q')).apply(
                lambda x: (1 + x).prod() - 1
            )
            return quarterly_ret
        elif freq == 'Y':
            # Yearly returns - group by year and compound
            yearly_ret = self.ret.groupby(self.ret.index.to_period('Y')).apply(
                lambda x: (1 + x).prod() - 1
            )
            return yearly_ret
        else:
            raise ValueError("freq must be 'D', 'M', 'Q', or 'Y'")

class Stats:
    def __init__(self, ret, rfr=0.00):
        self.ret = ret
        self.rfr = rfr

    def sharpe(self):
        res = (self.ret.mean() - self.rfr) * np.sqrt(252) / self.ret.std()
        return res

    def cumulative_return(self):
        res = (1 + self.ret).prod() - 1
        return res
    
    def annualized_return(self):
        years = len(self.ret) / 252
        res = (1 + self.ret).prod() ** (1 / years) - 1
        return res
    
    def annualized_volatility(self):
        res = self.ret.std() * np.sqrt(252)
        return res
    
    def sortino(self):
        downside = (self.ret[self.ret < 0] ** 2).sum() / len(self.ret)
        res = (self.ret.mean() - self.rfr) * np.sqrt(252) / np.sqrt(downside)
        return res
    
    def beta(self, benchmark):
        benchmark = benchmark.loc[benchmark.index.isin(self.ret.index)]
        res = self.ret.cov(benchmark) / benchmark.var()
        return res
    
    def alpha(self, benchmark):
        benchmark = benchmark.loc[benchmark.index.isin(self.ret.index)]
        benchmark_annualized_return = (1 + benchmark).prod() ** (252 / len(benchmark)) - 1
        res = self.annualized_return() - self.beta(benchmark) * benchmark_annualized_return
        return res
    
    def max_drawdown(self):
        ret = (1 + self.ret).cumprod()
        res = (ret / ret.expanding().max()).min() - 1
        return res
    
    def longest_drawdown(self):
        ret = (1 + self.ret).cumprod()
        drawdown = ret / ret.cummax() - 1
        is_drawdown = drawdown < 0
        group = (is_drawdown != is_drawdown.shift(1)).cumsum()
        drawdown_groups = group[is_drawdown]
        res = drawdown_groups.value_counts().max()
        return res
    
    def calmar(self):
        res = self.annualized_return() / abs(self.max_drawdown())
        return res
    
    def skew(self):
        res = self.ret.skew()
        return res
    
    def kurt(self):
        res = self.ret.kurt()
        return res
    
    

class Plots:
    def __init__(self, ret):
        self.ret = ret
    
    def ret_graph(self, benchmark=None, show_excess=False, show_drawdown=False):

        res = self.ret.to_cumulative_value()
        
        # Create Figure
        if show_drawdown:
            fig, axes = plt.subplots(
                2, 1,
                figsize=(12, 8),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0}
            )
        else:
            fig, axes = plt.subplots(figsize=(12, 6))
            axes = np.array([axes])
        
        fig.set_facecolor('white')
        fig.suptitle(self.ret.name + ' Performance', fontsize=16)
        
        # Plot strategy returns
        sns.lineplot(
            data=res,
            ax=axes[0],
            label="Strategy",
            color='blue',
            alpha=0.8
        )
        
        # Plot benchmark if provided
        if benchmark is not None:
            if hasattr(benchmark, 'to_cumulative_value'):
                benchmark_cum = benchmark.to_cumulative_value()
            else:
                benchmark_cum = (1 + benchmark).cumprod()
            
            sns.lineplot(
                data=benchmark_cum,
                ax=axes[0],
                label="Benchmark",
                color='red',
                alpha=0.8
            )
            
            # Show excess cumulative value if requested
            if show_excess:
                excess_ret = self.ret - benchmark
                excess_cum = (1 + excess_ret).cumprod()

                axes[0].fill_between(
                    excess_cum.index,
                    excess_cum,
                    0,
                    color='pink',
                    alpha=0.25,
                    label="Excess"
                )
        
        # Configure main plot
        axes[0].set_ylabel('Cumulative Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot drawdown if requested
        if show_drawdown:
            dd = self.ret.to_drawdown()
            sns.lineplot(
                data=dd,
                ax=axes[1],
                label="Drawdown",
                color='gray',
                alpha=0.8
            )
            axes[1].fill_between(
                dd.index, dd, 0, color='gray', alpha=0.25
            )
            
            # Format y-axis as percentage
            def percentage_formatter(x, pos):
                return f'{x:.0%}'
            
            axes[1].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
            axes[1].set_ylabel('Drawdown')
            axes[1].set_xlabel('Date')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[0].set_xlabel('Date')
        
        plt.tight_layout()
        return fig
    
    def hist(self, freq='D'):

        res = self.ret.to_return(freq)
        
        # Create Figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        fig.suptitle(f"Distribution of Returns ({freq})", fontsize=16)
        
        # Calculate number of bins
        n_bins = min(50, max(5, int(len(res) ** 0.75)))
        
        # Plot KDE
        sns.kdeplot(
            data=res,
            color='black',
            ax=ax,
            warn_singular=False,
            label="KDE"
        )
        
        # Plot histogram
        sns.histplot(
            data=res,
            bins=n_bins,
            alpha=0.8,
            kde=False,
            stat="density",
            color='skyblue',
            ax=ax,
            label="Strategy"
        )
        
        # Add mean line
        ax.axvline(
            res.mean(), 
            ls="--", 
            lw=2, 
            color="green", 
            zorder=2, 
            alpha=0.5, 
            label="Mean"
        )
        
        # Format x-axis as percentage
        def percentage_formatter(x, pos):
            return f'{x:.1%}'
        
        ax.xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        ax.set_xlabel('Return')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def heatmap(self, freq='M'):

        # Get returns at specified frequency
        res = self.ret.to_return(freq)
        
        # Convert to DataFrame for processing
        res_df = pd.DataFrame({
            'returns': res.values,
            'year': res.index.strftime("%Y"),
            'period': res.index.strftime("%m") if freq == 'M' else res.index.quarter
        }, index=res.index)
        
        # Set column name based on frequency
        if freq == 'M':
            name = "Month"
        elif freq == 'Q':
            name = "Quarter"
        else:
            raise ValueError("freq must be 'M', 'Q' ")
        
        # Pivot the data
        pivot_data = res_df.pivot(index="year", columns="period", values="returns").fillna(0)
        
        # Sort columns for better visualization
        if freq == 'M':
            pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)
        elif freq == 'Q':
            pivot_data = pivot_data.reindex(sorted(pivot_data.columns), axis=1)
        
        # Create Figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        fig.suptitle(f"Returns Heatmap ({name})", fontsize=16)
        
        # Plot heatmap
        sns.heatmap(
            pivot_data,
            ax=ax,
            annot=True,
            center=0,
            annot_kws={"size": 10},
            fmt=".2%",
            linewidths=0.5,
            cbar=True,
            cmap="RdYlGn_r",
            cbar_kws={"format": FuncFormatter(lambda x, p: f"{x:.2%}")}
        )
        
        ax.set_xlabel(name)
        ax.set_ylabel('Year')
        
        plt.tight_layout()
        return fig
    
    def rolling_beta(self, benchmark, window='3M'):

        if window == '3M':
            periods = 63  # ~3 months of trading days
        elif window == '6M':
            periods = 126  # ~6 months of trading days
        elif window == '1Y':
            periods = 252  # ~1 year of trading days
        else:
            raise ValueError("window must be '3M', '6M', or '1Y'")
        
        # Align data
        aligned_ret = self.ret.loc[self.ret.index.isin(benchmark.index)]
        aligned_benchmark = benchmark.loc[benchmark.index.isin(self.ret.index)]
        
        # Calculate rolling beta using vectorized operations
        rolling_cov = aligned_ret.rolling(window=periods).cov(aligned_benchmark)
        rolling_var = aligned_benchmark.rolling(window=periods).var()
        rolling_beta = rolling_cov / rolling_var
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        fig.suptitle(f"Rolling Beta (Window: {window})", fontsize=16)
        
        ax.plot(rolling_beta.index, rolling_beta.values, color='blue', alpha=0.8, linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_ylabel('Beta')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def rolling_sharpe(self, window='3M', rfr=0.00):

        if window == '3M':
            periods = 63
        elif window == '6M':
            periods = 126
        elif window == '1Y':
            periods = 252
        else:
            raise ValueError("window must be '3M', '6M', or '1Y'")
        
        # Calculate rolling Sharpe ratio using vectorized operations
        rolling_mean = self.ret.rolling(window=periods).mean()
        rolling_std = self.ret.rolling(window=periods).std()
        rolling_sharpe = (rolling_mean - rfr) * np.sqrt(252) / rolling_std
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        fig.suptitle(f"Rolling Sharpe Ratio (Window: {window})", fontsize=16)
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', alpha=0.8, linewidth=1.5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax.set_ylabel('Sharpe Ratio')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def rolling_volatility(self, window='1Y'):

        # Convert time window to number of periods
        if window == '3M':
            periods = 63
        elif window == '6M':
            periods = 126
        elif window == '1Y':
            periods = 252
        else:
            raise ValueError("window must be '3M', '6M', or '1Y'")
        
        # Calculate rolling volatility using vectorized operations
        rolling_vol = self.ret.rolling(window=periods).std() * np.sqrt(252)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        fig.suptitle(f"Rolling Volatility (Window: {window})", fontsize=16)
        
        ax.plot(rolling_vol.index, rolling_vol.values, color='purple', alpha=0.8, linewidth=1.5)
        
        # Format y-axis as percentage
        def percentage_formatter(x, pos):
            return f'{x:.1%}'
        
        ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        ax.set_ylabel('Volatility')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def rolling_sortino(self, window='6M', rfr=0.00):

        if window == '3M':
            periods = 60
        elif window == '6M':
            periods = 126
        elif window == '1Y':
            periods = 252
        else:
            raise ValueError("window must be '3M', '6M', or '1Y'")
        
        # Calculate rolling Sortino ratio using vectorized operations
        rolling_mean = self.ret.rolling(window=periods).mean()
        
        # Calculate downside deviation
        downside_returns = self.ret.copy()
        downside_returns[downside_returns >= 0] = 0
        rolling_downside_std = downside_returns.rolling(window=periods).std()
        
        rolling_sortino = (rolling_mean - rfr) * np.sqrt(252) / rolling_downside_std
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.set_facecolor('white')
        ax.set_facecolor('white')
        fig.suptitle(f"Rolling Sortino Ratio (Window: {window})", fontsize=16)
        
        ax.plot(rolling_sortino.index, rolling_sortino.values, color='orange', alpha=0.8, linewidth=1.5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax.set_ylabel('Sortino Ratio')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    
