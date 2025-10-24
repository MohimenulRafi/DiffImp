## Results ##
**Prediction Change Analysis**

<table>
  <tr>
    <th colspan="7">MIMIC III (Death to Survival)</th>
    <th colspan="7">MIMIC III (Survival to Death)</th>
  </tr>
  <tr>
    <th>Models</th>
    <th colspan="2">Correct</th>
    <th colspan="2">Incorrect</th>
    <th colspan="2">Survival Rate (Training Set)</th>
    <th colspan="2">Correct</th>
    <th colspan="2">Incorrect</th>
    <th colspan="2">Death Rate (Training Set)</th>
  </tr>
  <tr>
    <th></th><th>Male</th><th>Female</th><th>Male</th><th>Female</th><th>Male</th><th>Female</th>
    <th>Male</th><th>Female</th><th>Male</th><th>Female</th><th>Male</th><th>Female</th>
  </tr>
  <tr><td>BERT</td><td>3.17</td><td>3.54</td><td>0.98</td><td>1.07</td><td></td><td></td><td>1.28</td><td>1.76</td><td>7.03</td><td>7.32</td><td></td><td></td></tr>
  <tr><td>LSTM</td><td>2.67</td><td>3.5</td><td>0.89</td><td>0.97</td><td>88.86</td><td>87.44</td><td>0.59</td><td>0.35</td><td>1.34</td><td>1.07</td><td>11.14</td><td>12.56</td></tr>
  <tr><td>Transformer</td><td>2.69</td><td>5.07</td><td>1.76</td><td>2.29</td><td></td><td></td><td>1.22</td><td>1.23</td><td>3.47</td><td>2.71</td><td></td><td></td></tr>

  <tr><th colspan="7">eICU (Death to Survival)</th><th colspan="7">eICU (Survival to Death)</th></tr>
  <tr>
    <th>Models</th>
    <th colspan="2">Correct</th>
    <th colspan="2">Incorrect</th>
    <th colspan="2">Survival Rate (Training Set)</th>
    <th colspan="2">Correct</th>
    <th colspan="2">Incorrect</th>
    <th colspan="2">Death Rate (Training Set)</th>
  </tr>
  <tr>
    <th></th><th>Male</th><th>Female</th><th>Male</th><th>Female</th><th>Male</th><th>Female</th>
    <th>Male</th><th>Female</th><th>Male</th><th>Female</th><th>Male</th><th>Female</th>
  </tr>
  <tr><td>BERT</td><td>6.1</td><td>6.97</td><td>1.63</td><td>1.4</td><td></td><td></td><td>0.47</td><td>0.6</td><td>3.36</td><td>2.93</td><td></td><td></td></tr>
  <tr><td>LSTM</td><td>1.43</td><td>2.77</td><td>0.53</td><td>0.6</td><td>88.73</td><td>88.31</td><td>0.43</td><td>0.35</td><td>1.71</td><td>1</td><td>11.27</td><td>11.69</td></tr>
  <tr><td>Transformer</td><td>2.98</td><td>2.77</td><td>1.36</td><td>0.9</td><td></td><td></td><td>0.76</td><td>0.97</td><td>2.42</td><td>5.28</td><td></td><td></td></tr>
</table>

**Table 4: Percentages of change in prediction for gender groups. The table shows the percentages of correct and incorrect prediction changes by the models from death to survival and from survival to death in MIMIC III and eICU.**

<br />
