import numpy as np
import pandas as pd
import plotly.express as px
import os

def generate_visual_analysis(file_path):
    """
    Generates an interactive Parallel Coordinates plot to analyze the 
    Cost Function distribution and scale.
    Args:
        file_path: Path to the .npz dataset file containing 'inputs' and 'targets'.
    Returns:
        None (opens an interactive plot in the default web browser)
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    print(f"Loading data from: {file_path}...")
    data = np.load(file_path)
    
    try:
        x = data['inputs']   # State vector [q1, q2, dq1, dq2]
        y = data['targets']  # Cost / Value function values
    except KeyError as e:
        print(f"Error: Key not found. Available keys in file: {list(data.keys())}")
        return
    
    # Ensure the cost array is 1D for DataFrame compatibility
    y = y.flatten()

    # 1. Create the DataFrame
    df = pd.DataFrame(x, columns=['q1', 'q2', 'dq1', 'dq2'])
    df['Cost'] = y

    # 2. Logarithmic Transformation for the color scale
    # othrwise small values would have same color as zero essentially
    # (Just for the colors vis)
    df['Log_Cost'] = np.log10(df['Cost'] + 1e-9)

    # 3. Generate Interactive Parallel Coordinates Plot
    print("Generating interactive plot...")
    fig = px.parallel_coordinates(
        df, 
        dimensions=['q1', 'q2', 'dq1', 'dq2', 'Cost'],
        color="Log_Cost", 
        color_continuous_scale=px.colors.sequential.Turbo,
        labels={
            "q1": "q1 (rad)",
            "q2": "q2 (rad)",
            "dq1": "dq1 (rad/s)",
            "dq2": "dq2 (rad/s)",
            "Cost": "Cost Value"
        },
        title="Cost Function Analysis: Scale and Distribution"
    )

    # Clean up layout margins
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Cost (10^x)", # Explicitly stating it's a power of 10
            ticks="outside"
        ),
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    print("Opening plot in your default browser...")
    fig.show()


# Main execution
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, 'pendulum_dataset_used.npz')
    
    generate_visual_analysis(path)