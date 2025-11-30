import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_replication_results(values, output_file='replication_results.html'):
    """
    Plots a sequence of values as points connected by a dotted line and saves the
    figure to `output_file`.

    Args:
        values (list[float]): Sequence of numeric values to plot. Each point's x
            coordinate will be its index in the list plus one (1-based).
        output_file (str): Path to save the plot. If the extension is `.html` the
            plot will be saved as interactive HTML. For other extensions the
            function will attempt to save an image (requires Plotly image engine
            like `kaleido`); if that fails a `.html` fallback will be written.

    Returns:
        str: Path to the file that was written (the image path or the HTML fallback).
    """

    if not isinstance(values, (list, tuple)):
        raise TypeError('values must be a list or tuple of numbers')

    x1 = list(range(1, len(values[0]) + 1))
    x2 = list(range(1, len(values[1]) + 1))

    canvas = make_subplots(rows=1, cols=2, subplot_titles=('Math Accuracy vs Number of Debating Agents','Math Accuracy vs Debate Rounds'))

    canvas.add_trace(go.Scatter(
        x=x1,
        y=values[0],
        mode='lines+markers',
        line=dict(dash='dot', color='black'),
        marker=dict(size=8)
    ),
    row=1, 
    col=1)

    canvas.add_trace(go.Scatter(
        x=x2,
        y=values[1],
        mode='lines+markers',
        line=dict(dash='dot', color='black'),
        marker=dict(size=8)
    ),
    row=1, 
    col=2)

    canvas.update_xaxes(row=1, col=2, tickmode='linear', dtick=1, tick0=1)
    

    canvas.update_layout(
        template='simple_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    canvas.update_xaxes(
        title_text='Agent Number',
        showgrid=True,
        gridcolor='lightgrey',
        gridwidth=1,
        showline=True,
        linecolor='black',
        linewidth=1,
        mirror=True,
    )
    canvas.update_yaxes(
        title_text='Math Accuracy',
        showgrid=True,
        gridcolor='lightgrey',
        gridwidth=1,
        showline=True,
        linecolor='black',
        linewidth=1,
        mirror=True,
    )


    # Ensure output directory exists
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save: if HTML requested, write HTML. Otherwise try to write an image and
    # fall back to HTML if the image write fails (kaleido/orca missing).
    ext = os.path.splitext(output_file)[1].lower()
    try:
        if ext == '.html' or ext == '':
            canvas.write_html(output_file)
            return output_file
        else:
            canvas.write_image(output_file)
            return output_file
    except Exception as e:
        # Fallback: save HTML next to the requested path
        fallback = output_file + '.html' if not output_file.endswith('.html') else output_file
        canvas.write_html(fallback)
        print(f'Warning: could not write image ({e}). Saved HTML fallback to {fallback}')
        return fallback
    
def main():

    rounds_2_data = [0.82, 0.88, 0.96, 0.97, 0.98, 0.99, 0.97]
    rounds_3_data = [0.9, 0.93, 0.93, 0.94]

    values = [rounds_2_data, rounds_3_data]

    plot_replication_results(values, 'replication_results.html')

if __name__ == '__main__':
    main()

