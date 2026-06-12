%% DDM Lab 1 - Model Order Selection
% Runs the FIR model-order selection experiments for Lab 1.
%
% The script builds a known Chebyshev type II system, simulates noisy input
% and output data, estimates FIR models with several candidate orders, and
% compares three ways of choosing the order:
%   - least-squares training cost,
%   - Akaike's information criterion (AIC),
%   - validation cost on a separate dataset.


clear; close all; clc;
rng(7, "twister"); %set the random seed of the runs 

outputDir = fullfile(pwd, "figures", "lab1");
if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

baseConfig = struct( ...
    "filterOrder", 2, ...
    "stopbandAttenuationDb", 3, ...
    "bandEdges", [0.3 0.6], ...
    "impulseLength", 30, ...
    "dataLengthFactor", 10, ...
    "maxModelOrder", 100, ...
    "monteCarloRuns", 100);

% The three lab cases use the same workflow but change either the SNR or the
% bandwidth of the underlying system.
scenarios = [
    struct("name", "normal_snr", "title", "Normal SNR, 6 dB", ...
        "snrDb", 6, "bandEdges", [0.3 0.6], "runMonteCarlo", true)
    struct("name", "high_snr", "title", "High SNR, 26 dB", ...
        "snrDb", 26, "bandEdges", [0.3 0.6], "runMonteCarlo", true)
    struct("name", "narrow_bandwidth", "title", "Reduced Bandwidth, 6 dB", ...
        "snrDb", 6, "bandEdges", [0.4 0.5], "runMonteCarlo", false)
];

summary = table();
for k = 1:numel(scenarios)
    % Start from the shared settings, then apply the values for this case.
    config = baseConfig;
    config.bandEdges = scenarios(k).bandEdges;
    config.snrDb = scenarios(k).snrDb;

    % Run one complete estimation/validation pass and add it to the summary.
    result = runOrderSelectionScenario(config);
    row = scenarioSummaryRow(scenarios(k), result);
    if isempty(summary)
        summary = row;
    else
        summary = [summary; row]; %#ok<AGROW>
    end

    scenarioDir = fullfile(outputDir, scenarios(k).name);
    if ~exist(scenarioDir, "dir")
        mkdir(scenarioDir);
    end

    plotCostCurves(result, scenarios(k), scenarioDir);
    plotCostCurvesZoom(result, scenarios(k), scenarioDir);

    if scenarios(k).runMonteCarlo
        % Repeat the noisy experiment to show how stable AIC and validation
        % are when the input and noise realization change.
        robust = runRobustnessExperiment(config);
        plotRobustnessHistograms(robust, result.trueOrder, scenarios(k), scenarioDir);
    end
end

disp("=== Single-Run Model Order Selection Summary ===");
disp(summary);

save(fullfile(outputDir, "lab1_results.mat"), "summary", "baseConfig", "scenarios");

%% Local functions
function result = runOrderSelectionScenario(config)
%RUN ORDER SELECTION SCENARIO Estimate all candidate FIR orders for one case.
%   The returned structure contains the generated data, the training/AIC/
%   validation costs, the selected orders, and the measured SNR values. This
%   is the main numerical workflow used by each scenario in the script.

    [g0, impulseFull] = buildExactModel(config);

    % The lab treats the truncated impulse response length as the true FIR
    % order. Estimation and validation use equally sized datasets.
    trueOrder = config.impulseLength;
    ne = config.dataLengthFactor * trueOrder;
    nv = ne;

    % Use one dataset to estimate theta and an independent dataset to check
    % how well each estimated model predicts new data.
    [ue, ye0, yme, sigmaE] = simulateDataset(g0, ne, config.snrDb);
    [uv, yv0, ymv, sigmaV] = simulateDataset(g0, nv, config.snrDb);

    [theta, vLs] = estimateAllOrders(ue, yme, sigmaE, config.maxModelOrder);

    % AIC keeps the training fit but adds a penalty for using more
    % parameters, so it is less eager to pick very large models.
    vAic = vLs .* (1 + 2 * (1:config.maxModelOrder)' / ne);
    vVal = validationCost(uv, ymv, sigmaV, theta, config.maxModelOrder);

    % The best order is the index of the smallest cost for each criterion.
    [~, nOptLs] = min(vLs);
    [~, nOptAic] = min(vAic);
    [~, nOptVal] = min(vVal);

    result = struct( ...
        "g0", g0, ...
        "impulseFull", impulseFull, ...
        "trueOrder", trueOrder, ...
        "ne", ne, ...
        "nv", nv, ...
        "ue", ue, ...
        "uv", uv, ...
        "ye0", ye0, ...
        "yv0", yv0, ...
        "yme", yme, ...
        "ymv", ymv, ...
        "sigmaE", sigmaE, ...
        "sigmaV", sigmaV, ...
        "vLs", vLs, ...
        "vAic", vAic, ...
        "vVal", vVal, ...
        "nOptLs", nOptLs, ...
        "nOptAic", nOptAic, ...
        "nOptVal", nOptVal, ...
        "actualSnrE", snrDb(ye0, yme - ye0), ...
        "actualSnrV", snrDb(yv0, ymv - yv0));
end

function [g0, impulseFull] = buildExactModel(config)
%BUILD EXACT MODEL Create the reference system used to generate the data.
%   A Chebyshev type II digital filter is built from the lab settings. The
%   full impulse response is kept for plotting/checking, and g0 is the
%   truncated impulse response that acts as the true FIR model.

    [b, a] = cheby2(config.filterOrder, config.stopbandAttenuationDb, config.bandEdges);
    sys = tf(b, a, 1);
    impulseFull = impulse(sys, 100);
    g0 = impulseFull(1:config.impulseLength);
end

function [u, y0, ym, sigmaNoise] = simulateDataset(g0, nSamples, targetSnrDb)
%SIMULATE DATASET Generate one noisy input/output experiment.
%   u is white Gaussian input. y0 is the noise-free system output. ym is the
%   measured output after adding Gaussian noise scaled to the requested SNR.
%   sigmaNoise is returned because the cost functions are normalized by the
%   noise variance.

    u = randn(nSamples, 1);
    y0 = filter(g0, 1, u);
    signalPower = mean(y0.^2);
    sigmaNoise = sqrt(signalPower / 10^(targetSnrDb / 10));
    ym = y0 + sigmaNoise * randn(nSamples, 1);
end

function [theta, vLs] = estimateAllOrders(u, y, sigmaNoise, maxOrder)
%ESTIMATE ALL ORDERS Fit FIR models from order 1 up to maxOrder.
%   The Toeplitz matrix contains the delayed input samples used for the FIR
%   regression. For each candidate order, the function solves a least-squares
%   problem and stores the normalized training error.

    nSamples = numel(u);
    hMax = toeplitz(u, [u(1); zeros(maxOrder - 1, 1)]');
    theta = cell(maxOrder, 1);
    vLs = zeros(maxOrder, 1);

    for n = 1:maxOrder
        % Keep only the first n columns so the regression has n FIR taps.
        hN = hMax(:, 1:n);
        theta{n} = hN \ y;
        residual = y - hN * theta{n};
        vLs(n) = (residual' * residual) / (nSamples * sigmaNoise^2);
    end
end

function vVal = validationCost(u, y, sigmaNoise, theta, maxOrder)
%VALIDATION COST Measure each fitted model on independent data.
%   The validation dataset is converted to the same FIR regression form as
%   the estimation data. Each stored theta{n} is tested on this new dataset,
%   which helps detect overfitting that is hidden in the training error.

    nSamples = numel(u);
    hMax = toeplitz(u, [u(1); zeros(maxOrder - 1, 1)]');
    vVal = zeros(maxOrder, 1);

    for n = 1:maxOrder
        % Reuse the coefficients estimated from the training data.
        hN = hMax(:, 1:n);
        residual = y - hN * theta{n};
        vVal(n) = (residual' * residual) / (nSamples * sigmaNoise^2);
    end
end

function robust = runRobustnessExperiment(config)
%RUN ROBUSTNESS EXPERIMENT Repeat the experiment over many noise realizations.
%   Each run creates fresh estimation and validation data, then records which
%   model order is selected by LS, AIC, and validation. The resulting arrays
%   are used to plot histograms of order-selection stability.

    [g0, ~] = buildExactModel(config);
    ne = config.dataLengthFactor * config.impulseLength;
    nv = ne;
    maxOrder = config.maxModelOrder;
    nRuns = config.monteCarloRuns;

    nOptLs = zeros(nRuns, 1);
    nOptAic = zeros(nRuns, 1);
    nOptVal = zeros(nRuns, 1);

    for runIdx = 1:nRuns
        % New random data makes this a Monte Carlo check instead of a repeat
        % of the single scenario result.
        [ue, ~, yme, sigmaE] = simulateDataset(g0, ne, config.snrDb);
        [uv, ~, ymv, sigmaV] = simulateDataset(g0, nv, config.snrDb);

        [theta, vLs] = estimateAllOrders(ue, yme, sigmaE, maxOrder);
        vAic = vLs .* (1 + 2 * (1:maxOrder)' / ne);
        vVal = validationCost(uv, ymv, sigmaV, theta, maxOrder);

        [~, nOptLs(runIdx)] = min(vLs);
        [~, nOptAic(runIdx)] = min(vAic);
        [~, nOptVal(runIdx)] = min(vVal);
    end

    robust = struct("nOptLs", nOptLs, "nOptAic", nOptAic, "nOptVal", nOptVal);
end

function row = scenarioSummaryRow(scenario, result)
%SCENARIO SUMMARY ROW Convert one scenario result into a table row.
%   This table keeps the report output focused on the model orders selected
%   by LS, AIC, and validation for each scenario.

    row = table( ...
        string(scenario.name), scenario.snrDb, string(mat2str(scenario.bandEdges)), ...
        result.trueOrder, result.nOptLs, result.nOptAic, result.nOptVal, ...
        'VariableNames', {'scenario', 'targetSnrDb', 'bandEdges', 'trueOrder', ...
        'nOptLs', 'nOptAic', 'nOptVal'});
end

function value = snrDb(signal, noise)
%SNR DB Compute signal-to-noise ratio in decibels.
%   The function uses average signal and noise power, which matches how the
%   simulated noise level is set in simulateDataset.

    value = 10 * log10(mean(signal.^2) / mean(noise.^2));
end

%%%%%PLOTTING FUNCTIONS%%%%%%

function plotCostCurves(result, scenario, outputDir)
%PLOT COST CURVES Save the full LS, AIC, and validation cost curves.
%   The selected order for each criterion is marked on the plot, along with
%   the true order used to generate the data.

    n = 1:numel(result.vLs);
    fig = figure("Visible", "on", "Color", "w", "Position", [100 100 980 620]);
    plot(n, result.vLs, "b-", "LineWidth", 1.5); hold on;
    plot(n, result.vAic, "r--", "LineWidth", 1.5);
    plot(n, result.vVal, "Color", [0 0.55 0], "LineStyle", "-.", "LineWidth", 1.5);
    plot(result.nOptLs, result.vLs(result.nOptLs), "bo", "MarkerFaceColor", "b");
    plot(result.nOptAic, result.vAic(result.nOptAic), "ro", "MarkerFaceColor", "r");
    plot(result.nOptVal, result.vVal(result.nOptVal), "o", "Color", [0 0.55 0], ...
        "MarkerFaceColor", [0 0.55 0]);
    xline(result.trueOrder, "--k", "n_H = " + string(result.trueOrder), "LineWidth", 1.3);
    xlabel("Model order n");
    ylabel("Normalized cost");
    title("Model Order Selection Criteria - " + scenario.title);
    legend("V_{LS} training", "V_{AIC}", "V_{val} validation", ...
        "LS minimum", "AIC minimum", "Validation minimum", "True order", ...
        "Location", "northeast");
    grid on;
    exportgraphics(fig, fullfile(outputDir, "02_cost_curves.png"), "Resolution", 300);
end

function plotCostCurvesZoom(result, scenario, outputDir)
%PLOT COST CURVES ZOOM Save full and zoomed views of the cost curves.
%   The upper panel gives the complete sweep up to maxModelOrder. The lower
%   panel zooms in on the first 50 orders, where the meaningful comparison is
%   easier to see.

    n = 1:numel(result.vLs);
    zoomEnd = min(50, numel(n));

    fig = figure("Visible", "on", "Color", "w", "Position", [100 100 980 720]);
    tiledlayout(2, 1, "TileSpacing", "compact");

    nexttile;
    plot(n, result.vLs, "b-", "LineWidth", 1.4); hold on;
    plot(n, result.vAic, "r--", "LineWidth", 1.4);
    plot(n, result.vVal, "Color", [0 0.55 0], "LineStyle", "-.", "LineWidth", 1.4);
    xline(result.trueOrder, "--k", "n_H", "LineWidth", 1.2);
    xlabel("Model order n");
    ylabel("Cost");
    title("Full range - " + scenario.title);
    legend("V_{LS}", "V_{AIC}", "V_{val}", "True order", "Location", "northeast");
    grid on;

    nexttile;
    nZoom = 1:zoomEnd;
    plot(nZoom, result.vLs(nZoom), "b-", "LineWidth", 1.4); hold on;
    plot(nZoom, result.vAic(nZoom), "r--", "LineWidth", 1.4);
    plot(nZoom, result.vVal(nZoom), "Color", [0 0.55 0], "LineStyle", "-.", "LineWidth", 1.4);
    xline(result.trueOrder, "--k", "n_H", "LineWidth", 1.2);
    xlabel("Model order n");
    ylabel("Cost");
    title("Zoom on relevant orders");
    grid on;

    exportgraphics(fig, fullfile(outputDir, "03_cost_curves_zoom.png"), "Resolution", 300);
end

function plotRobustnessHistograms(robust, trueOrder, scenario, outputDir)
%PLOT ROBUSTNESS HISTOGRAMS Save Monte Carlo histograms of selected orders.
%   One histogram is created for each selection method so the spread of the
%   chosen orders can be compared against the true order.

    edges = 0.5:1:100.5;
    fig = figure("Visible", "on", "Color", "w", "Position", [100 100 980 800]);
    tiledlayout(3, 1, "TileSpacing", "compact");

    plotOneHistogram(robust.nOptLs, edges, trueOrder, "LS training");
    plotOneHistogram(robust.nOptAic, edges, trueOrder, "AIC");
    plotOneHistogram(robust.nOptVal, edges, trueOrder, "Validation");

    sgtitle("Robustness of Selected Model Order - " + scenario.title);
    exportgraphics(fig, fullfile(outputDir, "04_robustness_histograms.png"), "Resolution", 300);
end

function plotOneHistogram(values, edges, trueOrder, labelText)
%PLOT ONE HISTOGRAM Draw one selected-order histogram in the current tile.
%   values contains the selected order from each Monte Carlo run. The
%   vertical reference line marks the true order.

    nexttile;
    histogram(values, edges, "FaceAlpha", 0.85);
    xline(trueOrder, "--k", "n_H", "LineWidth", 1.2);
    xlim([0 100]);
    xlabel("Selected model order n");
    ylabel("Frequency");
    title(labelText);
    grid on;
end
