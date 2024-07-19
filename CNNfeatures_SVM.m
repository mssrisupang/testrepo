clear all;
% https://colab.research.google.com/drive/1HUUEFnY7053xePrYz3OSyWlxsBnTyk7X?usp=sharing

% Load data
trainData = load('dataTrainCNN.mat');
valData = load('dataValCNN.mat');

% Convert features and labels to double
X_train = double(trainData.dataTrain(:, 2:end));
y_train = double(trainData.dataTrain(:, 1))+1;

% Validation Data
X_val = double(valData.dataVal(:, 2:end));
y_val = double(valData.dataVal(:, 1))+1;

%% %% % Standardize training data
% Step 1: Handle Infinite Values
X_train(isinf(X_train)) = NaN;
X_val(isinf(X_val)) = NaN;

% Step 2: Impute Missing Values (if any)
meanTrain = mean(X_train, 'omitnan');
X_train = fillmissing(X_train, 'constant', meanTrain);
X_val = fillmissing(X_val, 'constant', meanTrain); % Use training data mean

% Step 3: Standardization - zero mean and unit variance
mu = mean(X_train);
sigma = std(X_train);
sigma(sigma == 0) = 1;  % Prevent division by zero

X_train = (X_train - mu) ./ sigma;
X_val = (X_val - mu) ./ sigma;

% Step 4: Min-Max Scaling (optional)
minVal = min(X_train);
maxVal = max(X_train);
range = maxVal - minVal;
range(range == 0) = 1;  % Avoid division by zero

X_train = (X_train - minVal) ./ range;
X_val = (X_val - minVal) ./ range;

% Final check for NaNs
assert(~any(isnan(X_train(:))), 'NaNs detected in training data');
assert(~any(isnan(X_val(:))), 'NaNs detected in validation data');


%% % Train the SVM classifier
svmModel = fitcecoc(X_train, y_train, 'Verbose', 2);


% % Predict using the SVM model
predictions = predict(svmModel, X_val);

% Calculate the accuracy
accuracy = sum(predictions == y_val) / numel(y_val);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% % Generate confusion matrix
C = confusionmat(y_val, predictions);

% Display the confusion matrix
disp('Confusion Matrix:');
disp(C);

%% % Parameters for SVM
boxConstraint = [0.1, 1, 10];
kernelScale = [0.1, 1, 10];
numFolds = 5;  % Example: 5-fold cross-validation

% Define a template for SVM
bestAccuracy = 0;
bestParams = struct('BoxConstraint', NaN, 'KernelScale', NaN);

% Grid search with cross-validation
for i = 1:length(boxConstraint)
    for j = 1:length(kernelScale)
        t = templateSVM('BoxConstraint', boxConstraint(i), 'KernelFunction', 'rbf', 'KernelScale', kernelScale(j));

        % Perform k-fold cross-validation
        cvModel = fitcecoc(X_train, y_train, 'Learners', t, 'KFold', numFolds, 'Verbose', 2);

        % Calculate average k-fold validation accuracy
        kFoldAccuracy = 1 - kfoldLoss(cvModel, 'LossFun', 'ClassifError');

        fprintf('k-Fold Accuracy with BoxConstraint = %.2f and KernelScale = %.2f: %.2f%%\n', ...
                boxConstraint(i), kernelScale(j), kFoldAccuracy * 100);

        % Check if this model has the best accuracy
        if kFoldAccuracy > bestAccuracy
            bestAccuracy = kFoldAccuracy;
            bestModel = cvModel;
            bestParams.BoxConstraint = boxConstraint(i);
            bestParams.KernelScale = kernelScale(j);
        end
    end
end

% Output the best results
fprintf('Best k-Fold Accuracy: %.2f%%\n', bestAccuracy * 100);
fprintf('Best Parameters: BoxConstraint = %.2f, KernelScale = %.2f\n', ...
        bestParams.BoxConstraint, bestParams.KernelScale);

%% 
t = templateSVM('BoxConstraint', 10.00, 'KernelFunction', 'rbf', 'KernelScale', 10.00);
finalModel = fitcecoc(X_train, y_train, 'Learners', t);
 
% predictions = predict(finalModel, X_val);
C = confusionmat(y_val, predictions);
disp('Confusion Matrix:');
disp(C);

% Calculate precision, recall, and F1 score
precision = diag(C) ./ sum(C, 2);
recall = diag(C) ./ sum(C, 1)';
f1_scores = 2 * (precision .* recall) ./ (precision + recall);
disp('Precision per class:');
disp(precision);
disp('Recall per class:');
disp(recall);
disp('F1 Scores per class:');
disp(f1_scores);
save('finalSVMModel.mat', 'finalModel');


%% % Assuming 'finalModel' is your trained model and 'X_val' is your validation data
[~, score] = predict(finalModel, X_val);  % SVM scores

numClasses = numel(unique(y_val));

figure;
for i = 1:numClasses
    % Prepare binary labels
    trueClass = (y_val == i);
    
    % Score for class i vs all others
    scoreClass = score(:, i);
    
    % Compute ROC curve
    [Xroc, Yroc, T, AUC] = perfcurve(trueClass, scoreClass, true);
    
    % Plot ROC curve
    subplot(2, ceil(numClasses / 2), i);
    plot(Xroc, Yroc);
    xlabel('False positive rate');
    ylabel('True positive rate');
    title(sprintf('Class %d ROC (AUC = %.2f)', i, AUC));
end

