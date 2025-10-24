using Plots
using PyPlot
using FFTW
using StatsBase
using Statistics
using ImageFiltering: Kernel, imfilter!

function plot_reconstruction_2d(X::AbstractMatrix, X̃::AbstractMatrix)
    @assert size(X, 2) == size(X̃, 2) == 2
    
    matplotlib = PyPlot.matplotlib
    matplotlib.rc("font", size=14)
    matplotlib.rc("axes", labelsize=14, titlesize=14)
    matplotlib.rc("xtick", labelsize=14)
    matplotlib.rc("ytick", labelsize=14)
    matplotlib.rc("legend", fontsize=14)
    
    fig = figure(figsize=(8, 6))
    PyPlot.plot(X[:, 1], X[:, 2], label="ground truth", linewidth=2, color="blue", alpha=0.5)
    PyPlot.plot(X̃[:, 1], X̃[:, 2], label="prediction", linewidth=2, color="red", alpha=0.9)
    xlabel("x")
    ylabel("y")
    legend()
    tight_layout()
    return fig
end

function plot_reconstruction_2d(X::AbstractArray{T, 3}, X̃::AbstractArray{T, 3}) where {T}
    @assert size(X, 2) == size(X̃, 2) == 2

    matplotlib = PyPlot.matplotlib
    matplotlib.rc("font", size=14)
    matplotlib.rc("axes", labelsize=14, titlesize=14)
    matplotlib.rc("xtick", labelsize=14)
    matplotlib.rc("ytick", labelsize=14)
    matplotlib.rc("legend", fontsize=14)
    
    fig = figure(figsize=(8, 6))
    @views PyPlot.plot(X[:, 1, 1], X[:, 2, 1], label="ground truth", linewidth=2, color="blue", alpha=0.5)
    @views PyPlot.plot(X̃[:, 1, 1], X̃[:, 2, 1], label="prediction", linewidth=2, color="red", alpha=0.9)
    @views PyPlot.plot(X[:, 1, 2:end], X[:, 2, 2:end], linewidth=2, color="blue", alpha=0.3)
    @views PyPlot.plot(X̃[:, 1, 2:end], X̃[:, 2, 2:end], linewidth=2, color="red", alpha=0.5)
    xlabel("x")
    ylabel("y")
    legend()
    tight_layout()
    return fig
end

function plot_reconstruction_3d(X::AbstractMatrix, X̃::AbstractMatrix)
    @assert size(X, 2) == size(X̃, 2) == 3
    
    matplotlib = PyPlot.matplotlib
    matplotlib.rc("font", size=14)
    matplotlib.rc("axes", labelsize=14, titlesize=14)
    matplotlib.rc("xtick", labelsize=14)
    matplotlib.rc("ytick", labelsize=14)
    matplotlib.rc("legend", fontsize=14)
    
    fig = figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:, 1], X[:, 2], X[:, 3], label="ground truth", linewidth=2, color="blue", alpha=0.5)
    ax.plot(X̃[:, 1], X̃[:, 2], X̃[:, 3], label="prediction", linewidth=2, color="red", alpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    tight_layout()
    return fig
end

function plot_reconstruction_3d(X::AbstractArray{T, 3}, X̃::AbstractArray{T, 3}) where {T}
    @assert size(X, 2) == size(X̃, 2) == 3

    matplotlib = PyPlot.matplotlib
    matplotlib.rc("font", size=14)
    matplotlib.rc("axes", labelsize=14, titlesize=14)
    matplotlib.rc("xtick", labelsize=14)
    matplotlib.rc("ytick", labelsize=14)
    matplotlib.rc("legend", fontsize=14)
    
    fig = figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    @views ax.plot(X[:, 1, 1], X[:, 2, 1], X[:, 3, 1], label="ground truth", linewidth=2, color="blue", alpha=0.5)
    @views ax.plot(X̃[:, 1, 1], X̃[:, 2, 1], X̃[:, 3, 1], label="prediction", linewidth=2, color="red", alpha=0.9)
    @views ax.plot(X[:, 1, 2:end], X[:, 2, 2:end], X[:, 3, 2:end], linewidth=2, color="blue", alpha=0.3)
    @views ax.plot(X̃[:, 1, 2:end], X̃[:, 2, 2:end], X̃[:, 3, 2:end], linewidth=2, color="red", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    tight_layout()
    return fig
end

function plot_reconstruction_series(X::AbstractMatrix, X̃::AbstractMatrix)
    @assert size(X, 2) == size(X̃, 2) == 1
    
    matplotlib = PyPlot.matplotlib
    matplotlib.rc("font", size=14)
    matplotlib.rc("axes", labelsize=14, titlesize=14)
    matplotlib.rc("xtick", labelsize=14)
    matplotlib.rc("ytick", labelsize=14)
    matplotlib.rc("legend", fontsize=14)
    
    t = 1:size(X, 1)
    fig = figure(figsize=(12, 6))
    PyPlot.plot(t, X[:, 1], label="ground truth", linewidth=2, color="blue", alpha=0.5)
    PyPlot.plot(t, X̃[:, 1], label="prediction", linewidth=2, color="red", alpha=0.9)
    xlabel("t")
    ylabel("a.u.")
    legend()
    tight_layout()
    return fig
end

function plot_reconstruction_multiple_series(
    X::AbstractMatrix,
    X̃::AbstractMatrix,
    n_plots::Int,
)
    @assert size(X, 2) == size(X̃, 2) >= n_plots
    
    matplotlib = PyPlot.matplotlib
    matplotlib.rc("font", size=14)
    matplotlib.rc("axes", labelsize=14, titlesize=14)
    matplotlib.rc("xtick", labelsize=14)
    matplotlib.rc("ytick", labelsize=14)
    matplotlib.rc("legend", fontsize=14)
    
    t = 1:size(X, 1)
    fig, axes = subplots(n_plots, 1, figsize=(12, 2*n_plots), sharex=true)
    
    for i = 1:n_plots
        ax = n_plots == 1 ? axes : axes[i]
        ax.plot(t, X[:, i], label="ground truth", linewidth=2, color="blue", alpha=0.5)
        ax.plot(t, X̃[:, i], label="prediction", linewidth=2, color="red", alpha=0.9)
        ax.set_ylabel("Series $i")
        if i == 1
            ax.legend()
        end
        if i == n_plots
            ax.set_xlabel("t")
        end
    end
    tight_layout()
    return fig
end

function plot_reconstruction(
    X_gen::AbstractArray{T, N},
    X::AbstractArray{T, N},
    save_path::String,
) where {T, N}
    if size(X, 2) == 3
        fig = plot_reconstruction_3d(X, X_gen)
    elseif size(X, 2) == 2
        fig = plot_reconstruction_2d(X, X_gen)
    elseif size(X, 2) == 1
        fig = plot_reconstruction_series(X, X_gen)
    elseif size(X, 2) >= 3
        n_plots = size(X, 2) > 5 ? 5 : size(X, 2)
        @views fig =
            plot_reconstruction_multiple_series(X[:, :, 1], X_gen[:, :, 1], n_plots)
    end
    PyPlot.savefig(save_path, dpi=300, bbox_inches="tight")
    PyPlot.close(fig)  # Close the figure after saving
end

"""
Evaulation plotting for attractor forecast using provided context to DynaMix
"""

function plot_3D_attractor(ground_truth, context, prediction; 
    lim_gen=2000, lim_pse=500, 
    smoothing_sigma=2.0f0)
    # Set font sizes using PyCall's direct access to matplotlib
    matplotlib = PyPlot.matplotlib
    matplotlib.rc("font", size=18)
    matplotlib.rc("axes", labelsize=18, titlesize=18)
    matplotlib.rc("xtick", labelsize=18)
    matplotlib.rc("ytick", labelsize=18)
    matplotlib.rc("legend", fontsize=18)

    CL_length = size(context, 1)

    fig = figure(figsize=(18, 4))

    # Define the total width and the width ratio
    total_width = 1.0
    width_ratios = [1.5, 2.5, 1.2]
    sum_ratio = sum(width_ratios)

    # Calculate normalized widths
    widths = width_ratios ./ sum_ratio

    # Calculate positions (left edges)
    left_positions = [0.0]
    for i in 1:2
        push!(left_positions, left_positions[end] + widths[i])
    end

    # 3D plot in first column - spans all rows
    ax3d = fig.add_axes([left_positions[1], 0.1, widths[1], 0.8], projection="3d")
    ax3d.plot(ground_truth[1:end, 1], ground_truth[1:end, 2], ground_truth[1:end, 3], 
    label="ground truth", linewidth=3, color="#2C3E50", alpha=0.5)
    ax3d.plot(context[1:end, 1], context[1:end, 2], context[1:end, 3], 
    label="context", linewidth=3, color="#2C3E50", alpha=0.9)
    ax3d.plot(prediction[1:end, 1], prediction[1:end, 2], prediction[1:end, 3], 
    label="prediction", linewidth=3, color="#FF4242", alpha=0.9)

    labels = ["x", "y", "z"]
    height = 0.25  # Height for each row
    for i in 1:3
        y_pos = 0.0 + (3-i) * (height+0.15)  # Top to bottom positioning
        ax_ts = fig.add_axes([left_positions[2]+0.02, y_pos, widths[2]-0.05, height])
        ax_ts.plot(CL_length:(CL_length + lim_gen - 1), ground_truth[1:lim_gen, i], 
        label="ground truth", linewidth=3, color="#2C3E50", alpha=0.5)
        ax_ts.plot(1:CL_length, context[:, i], 
        label="context", linewidth=3, color="#2C3E50", alpha=0.9)
        ax_ts.plot(CL_length:(CL_length + lim_gen - 1), prediction[1:lim_gen, i], 
        label="prediction", linewidth=3, color="#FF4242", alpha=0.9)
        ax_ts.set_ylabel(labels[i])
        if i == 3
            ax_ts.set_xlabel("Time \$t\$")
        end
        ax_ts.set_yticks([-2, 0, 2])
    end

    for i in 1:3
        y_pos = 0.0 + (3-i) * (height+0.15)  # Top to bottom positioning
        ax_ps = fig.add_axes([left_positions[3]+0.02, y_pos, widths[3]-0.05, height])
        ps = normalized_and_smoothed_power_spectrum(ground_truth[:, i:i], smoothing_sigma)
        ps_gen = normalized_and_smoothed_power_spectrum(prediction[:, i:i], smoothing_sigma)

        ax_ps.plot(ps[1:lim_pse], label="ground truth", linewidth=3, color="#2C3E50", alpha=0.5)
        ax_ps.plot(ps_gen[1:lim_pse], label="prediction", linewidth=3, color="#FF4242", alpha=0.9)
        if i == 3
            ax_ps.set_xlabel("frequency \$f\$")
        end
        ax_ps.set_yscale("log")
    end

    fig.tight_layout()

    return fig
end

"""
Evaluation plotting for time series forecasting using provided context to DynaMix
"""


function plot_TS_forecast(ground_truth, context, prediction, lim=1000)
    matplotlib = PyPlot.matplotlib
    matplotlib.rc("font", size=18)
    matplotlib.rc("axes", labelsize=18, titlesize=18)
    matplotlib.rc("xtick", labelsize=18)
    matplotlib.rc("ytick", labelsize=18)
    matplotlib.rc("legend", fontsize=18)

    context_time = -size(context, 1):-1
    forecast_time = 0:lim-1

    fig = figure(figsize=(14, 4))

    # Plot context
    PyPlot.plot(
        context_time, context, color="#2C3E50", linewidth=4, alpha=0.9, label="Context"
    )

    # Plot ground truth
    PyPlot.plot(
        forecast_time, ground_truth[1:lim,:], color="#2C3E50", linewidth=4, alpha=0.5, label="Ground Truth"
    )

    PyPlot.plot(
        forecast_time, prediction[1:lim,:], color="#FF4242", linewidth=4, alpha=0.9, label="DynaMix"
    )

    xlabel("Time")
    ylabel("Value")
    legend()
    tight_layout()

    return fig
end


"""
plot_2D_attractor(ground_truth, context, prediction)
"""

function plot_2D_attractor(context, prediction)
    matplotlib = PyPlot.matplotlib
    matplotlib.rc("font", size=18)
    matplotlib.rc("axes", labelsize=18, titlesize=18)
    matplotlib.rc("xtick", labelsize=18)
    matplotlib.rc("ytick", labelsize=18)
    matplotlib.rc("legend", fontsize=18)

    fig = figure(figsize=(6, 6))

    PyPlot.plot(context[:, 1], context[:, 2], linewidth=5, color="#2C3E50", alpha=0.9, label="context")
    PyPlot.plot(prediction[:, 1], prediction[:, 2], linewidth=5, color="#FF4242", alpha=0.9, label="prediction")

    # Configure plot appearance
    title("", fontsize=20)
    xlabel(L"x")
    ylabel(L"y")
    PyPlot.xticks([-1,0,1,2])
    PyPlot.yticks([-1,0,1,2,3])
    legend(fontsize=12,loc="upper right")
    tight_layout()

    return fig
end