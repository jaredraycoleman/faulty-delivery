import numpy as np
import pandas as pd
import plotly.express as px

NPOINTS = 1000

def main():
    times = np.linspace(0, 1, NPOINTS)
    dt = times[1] - times[0]
    p = 1/len(times) # probability of starter failing at each time step
    helper_start = np.array([0.0, 1.0])
    pursuer_start = np.copy(helper_start)
    starter_start = np.array([0.0, 0.0])

    rows = []
    for m in np.linspace(0, 1, NPOINTS): # helper ratios
        helper_pos = np.copy(helper_start)
        starter_pos = np.copy(starter_start)
        expected_delivery_time_helper = 0
        # expected_delivery_time_pursuer = 0
        for t in times:
            delay_helper = np.linalg.norm(helper_pos - starter_pos)
            expected_delivery_time_helper += (1+delay_helper) * p

            helper_pos += dt * (np.array([m, 0.0]) - helper_pos) / np.linalg.norm(np.array([m, 0.0]) - helper_pos)
            starter_pos += np.array([dt, 0.0])

        rows.append([m, 'helper', expected_delivery_time_helper])

    for r in np.linspace(0, 1, NPOINTS): # pursuit ratios
        pursuer_pos = np.copy(pursuer_start)
        starter_pos = np.copy(starter_start)
        expected_delivery_time_pursuer = 0
        for t in times:
            delay_pursuer = np.linalg.norm(pursuer_pos - starter_pos)
            expected_delivery_time_pursuer += (1+delay_pursuer) * p

            pursuit_point = np.array([starter_pos[0] + (1-starter_pos[0])*r, 0.0])
            pursuer_pos += dt * (pursuit_point - pursuer_pos) / np.linalg.norm(pursuit_point - pursuer_pos)
            starter_pos += np.array([dt, 0.0])

        rows.append([r, 'pursuer', expected_delivery_time_pursuer])
    df = pd.DataFrame(rows, columns=['m', 'agent', 'expected_delivery_time'])

    fig = px.line(
        df, x='m', y='expected_delivery_time',
        color='agent',
        template='simple_white',
        title='Expected Delivery Time of Helper vs m',
    )
    fig.write_image('expected.png')


    # Get minimum delivery time and m for each agent
    min_delivery_time_helper = df[df['agent'] == 'helper']['expected_delivery_time'].min()
    min_delivery_time_pursuer = df[df['agent'] == 'pursuer']['expected_delivery_time'].min()
    m_helper = df[(df['agent'] == 'helper') & (df['expected_delivery_time'] == min_delivery_time_helper)]['m'].values[0]
    r_pursuer = df[(df['agent'] == 'pursuer') & (df['expected_delivery_time'] == min_delivery_time_pursuer)]['m'].values[0]

    print(f"Minimum expected delivery time for helper: {min_delivery_time_helper:0.4f} at m = {m_helper:0.4f}")
    print(f"Minimum expected delivery time for pursuer: {min_delivery_time_pursuer:0.4f} at r = {r_pursuer:0.4f}")




if __name__ == '__main__':
    main()
    
