## Dataset Information

<table>
  <tr>
    <th>Variable</th>
    <th>Type</th>
  </tr>
  <tr><td>Capillary refill rate</td><td>Categorical</td></tr>
  <tr><td>Diastolic blood pressure</td><td>Continuous</td></tr>
  <tr><td>Fraction inspired oxygen</td><td>Continuous</td></tr>
  <tr><td>Glasgow coma scale eye-opening</td><td>Categorical</td></tr>
  <tr><td>Glasgow coma scale motor response</td><td>Categorical</td></tr>
  <tr><td>Glasgow coma scale total</td><td>Categorical</td></tr>
  <tr><td>Glasgow coma scale verbal response</td><td>Categorical</td></tr>
  <tr><td>Glucose</td><td>Continuous</td></tr>
  <tr><td>Heart Rate</td><td>Continuous</td></tr>
  <tr><td>Height</td><td>Continuous</td></tr>
  <tr><td>Mean blood pressure</td><td>Continuous</td></tr>
  <tr><td>Oxygen saturation</td><td>Continuous</td></tr>
  <tr><td>Respiratory rate</td><td>Continuous</td></tr>
  <tr><td>Systolic blood pressure</td><td>Continuous</td></tr>
  <tr><td>Temperature</td><td>Continuous</td></tr>
  <tr><td>Weight</td><td>Continuous</td></tr>
  <tr><td>pH</td><td>Continuous</td></tr>
</table>

**Table 2: List of variables used for training and testing the models.**

<br />

<table>
  <tr>
    <th colspan="5">MIMIC III (Gender)</th>
    <th colspan="8">MIMIC III (Race)</th>
  </tr>
  <tr>
    <th>Split</th>
    <th colspan="2">Survive</th>
    <th colspan="2">Death</th>
    <th colspan="4">Survive</th>
    <th colspan="4">Death</th>
  </tr>
  <tr>
    <th></th><th>Male</th><th>Female</th><th>Male</th><th>Female</th>
    <th>Asian</th><th>White</th><th>Black</th><th>Hispanic</th>
    <th>Asian</th><th>White</th><th>Black</th><th>Hispanic</th>
  </tr>
  <tr><td>Train</td><td>88.86</td><td>87.44</td><td>11.14</td><td>12.56</td><td>87.81</td><td>88.95</td><td>93.04</td><td>92.62</td><td>12.19</td><td>11.05</td><td>6.96</td><td>7.38</td></tr>
  <tr><td>Validation</td><td>88.23</td><td>88.79</td><td>11.77</td><td>11.21</td><td>93.24</td><td>88.66</td><td>92.53</td><td>95.65</td><td>6.76</td><td>11.34</td><td>7.47</td><td>4.35</td></tr>
  <tr><td>Test</td><td>91.09</td><td>89.71</td><td>8.91</td><td>10.29</td><td>87.93</td><td>91.02</td><td>92.75</td><td>95.00</td><td>12.07</td><td>8.98</td><td>7.25</td><td>5.00</td></tr>

  <tr><th colspan="5">eICU (Gender)</th><th colspan="8">eICU (Race)</th></tr>
  <tr>
    <th>Split</th>
    <th colspan="2">Survive</th>
    <th colspan="2">Death</th>
    <th colspan="4">Survive</th>
    <th colspan="4">Death</th>
  </tr>
  <tr>
    <th></th><th>Male</th><th>Female</th><th>Male</th><th>Female</th>
    <th>Asian</th><th>White</th><th>Black</th><th>Hispanic</th>
    <th>Asian</th><th>White</th><th>Black</th><th>Hispanic</th>
  </tr>
  <tr><td>Train</td><td>88.73</td><td>88.31</td><td>11.27</td><td>11.69</td><td>87.68</td><td>88.31</td><td>90.22</td><td>88.55</td><td>12.32</td><td>11.69</td><td>9.78</td><td>11.45</td></tr>
  <tr><td>Validation</td><td>88.21</td><td>88.42</td><td>11.79</td><td>11.58</td><td>88.61</td><td>87.79</td><td>90.86</td><td>90.38</td><td>11.39</td><td>12.21</td><td>9.14</td><td>9.62</td></tr>
  <tr><td>Test</td><td>88.35</td><td>88.94</td><td>11.65</td><td>11.06</td><td>90.28</td><td>88.29</td><td>90.91</td><td>87.20</td><td>9.72</td><td>11.71</td><td>9.09</td><td>12.80</td></tr>
</table>

**Table 3: Percentages of survival and death in data splits for gender and race groups.**

<br />

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

