
import heapq
import random
import numbers
import numpy as np
from sklearn.metrics import average_precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(-1,input.size(2))   # N,L,C => N*L,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def get_ap(gts_raw, preds_raw):
    gts, preds = gts_raw.reshape(-1).tolist(), preds_raw.reshape(-1).tolist()
    # print ("AP ",average_precision_score(gts, preds))
    return average_precision_score(np.nan_to_num(gts), np.nan_to_num(preds))


def get_mAP(loader, gts_raw, preds_raw):
    mAP, gts, preds = [], [], []
    for gt_raw in gts_raw:
        gts.extend(gt_raw.tolist())
    for pred_raw in preds_raw:
        preds.extend(pred_raw.tolist())

    n = min(len(loader.dataset), len(gts), len(preds))
    lines = []
    for i in range(n):
        one_idx = loader.dataset.listIDs[i]
        line = '{} {} {} {}'.format(one_idx['imdbid'], one_idx['shotid'],
                                    gts[i], preds[i])
        lines.append(line)
    lines = list(sorted(lines))
    imdbs = np.array([x.split(' ')[0] for x in lines])
    # shots = np.array([x.split(' ')[1] for x in lines])
    gts = np.array([int(x.split(' ')[2]) for x in lines], np.int32)
    preds = np.array([float(x.split(' ')[3]) for x in lines], np.float32)
    movies = np.unique(imdbs)
    for movie in movies:
        index = np.where(imdbs == movie)[0]
        ap = average_precision_score(gts[index], preds[index])
        mAP.append(round(ap, 2))
    return np.mean(mAP), np.array(mAP)


### interval_1 [start, end]   interval_2 [start, end]
def getIntersection(interval_1, interval_2):
#     if not interval_1[0] <= interval_1[1]:
#         print(interval_1)
#     if not interval_2[0] <= interval_2[1]:
#         print(interval_2)
#     assert interval_1[0] <= interval_1[1], "start frame is bigger than end frame. %s" % str(interval_1)
#     assert interval_2[0] <= interval_2[1], "start frame is bigger than end frame. %s" % str(interval_2)
    start = max(interval_1[0], interval_2[0])
    end = min(interval_1[1], interval_2[1])
    if start < end:
        return (end - start)
    return 0


def getUnion(interval_1, interval_2):
#     if not interval_1[0] <= interval_1[1]:
#         print(interval_1)
#     if not interval_2[0] <= interval_2[1]:
#         print(interval_2)
#     assert interval_1[0] <= interval_1[1], "start frame is bigger than end frame."
#     assert interval_2[0] <= interval_2[1], "start frame is bigger than end frame."
    start = min(interval_1[0], interval_2[0])
    end = max(interval_1[1], interval_2[1])
    return (end - start)


def getRatio(interval_1,interval_2):
    interaction = getIntersection(interval_1,interval_2)
    if interaction == 0:
        return 0
    else:
        return interaction/getUnion(interval_1,interval_2)


### gt_scene_list [[start1, end1], [start2, end2]…]    pred_scene_list [[start1, end1], [start2, end2]…]
def cal_Miou(gt_scene_list, pred_scene_list):
    mious = []
    for gt_scene_item in gt_scene_list:
         rats = []
         for pred_scene_item in pred_scene_list:
              rat = getRatio(pred_scene_item,gt_scene_item)
              rats.append(rat)
         mious.append(np.max(rats))  
    miou = np.mean(mious) 
    return miou


def get_Miou(pred_scene_list, gt_scene_list, threshold=0.5):
    Mious = []
    for index, pair in enumerate(zip(pred_scene_list, gt_scene_list)):
       pred_scene, gt_scene = pair
       if pred_scene is None:
           Mious.append(0)
           continue
       if gt_scene is None or pred_scene is None:
           return None
       miou1 = cal_Miou(gt_scene, pred_scene)
       miou2 = cal_Miou(pred_scene, gt_scene)
       Mious.append(np.mean([miou1, miou2]))
#     print("Miou: ", np.mean(Mious))
    return np.mean(Mious)


def get_recall(pred_scene_np, gt_scene_np):
    gt_scene, pred_scene = gt_scene_np.reshape(-1).tolist(), pred_scene_np.reshape(-1).tolist()
    # print ("AP ",average_precision_score(gts, preds))
    return recall_score(np.nan_to_num(gt_scene), np.nan_to_num(pred_scene))


def get_mAP_seq(loader, gts_raw, preds_raw):
    mAP = []
    gts, preds = [], []
    for gt_raw in gts_raw:
        gts.extend(gt_raw.tolist())
    for pred_raw in preds_raw:
        preds.extend(pred_raw.tolist())

    seq_len = len(loader.dataset.listIDs[0])
    n = min(len(loader.dataset), len(gts) // seq_len, len(preds) // seq_len)
    lines = []
    for i in range(n):
        for j in range(seq_len):
            one_idx = loader.dataset.listIDs[i][j]
            line = '{} {} {} {}'.format(one_idx['imdbid'], one_idx['shotid'],
                                        gts[i * seq_len + j], preds[i * seq_len + j])
            lines.append(line)
    lines = list(sorted(lines))
    imdbs = np.array([x.split(' ')[0] for x in lines])
    # shots = np.array([x.split(' ')[1] for x in lines])
    gts = np.array([int(x.split(' ')[2]) for x in lines], np.int32)
    preds = np.array([float(x.split(' ')[3]) for x in lines], np.float32)
    movies = np.unique(imdbs)
    # print("movies:", movies)
    for movie in movies:
        index = np.where(imdbs == movie)[0]
        ap = average_precision_score(np.nan_to_num(gts[index]), np.nan_to_num(preds[index]))
        mAP.append(round(np.nan_to_num(ap), 2))
        # print(mAP)
    return np.mean(mAP), np.array(mAP)


def get_time_f1(pred_time_spans, gt_time_spans, mean=True):
    if mean:
        f1 = 0.0
        for pred, gt in zip(pred_time_spans, gt_time_spans):
            pred_points = [p[0] for p in pred][1:]
            gt_points = [p[0] for p in gt][1:]
            pp = len(pred_points)
            ap = len(gt_points)
            if pp == 0 and ap == 0:
                f1 += 1.0
            elif pp == 0 and ap != 0:
                f1 += 0.0
            elif pp != 0 and ap == 0:
                f1 += 0.0
            else:
                tp = 0
                while len(pred_points) > 0 and len(gt_points) > 0:
                    if abs(pred_points[0] - gt_points[0]) < 0.5:
                        tp += 1
                        gt_points = gt_points[1:]
                    pred_points = pred_points[1:]
                p = tp / pp
                r = tp / ap

                if p == r == 0.0:
                    f1 += 0.0
                else:
                    f1 += 2 * p * r / (p + r)

        return f1 / len(pred_time_spans)
    else:
        tp, pp, ap = 0, 0, 0
        for pred, gt in zip(pred_time_spans, gt_time_spans):
            pred_points = [p[0] for p in pred][1:]
            gt_points = [p[0] for p in gt][1:]
            pp += len(pred_points)
            ap += len(gt_points)
            while len(pred_points) > 0 and len(gt_points) > 0:
                if abs(pred_points[0] - gt_points[0]) < 0.5:
                    tp += 1
                    gt_points = gt_points[1:]
                pred_points = pred_points[1:]
        p = tp / pp
        r = tp / ap
        f1 = 2 * p * r / (p + r)

        return f1


def calculate_gap(predictions, actuals, top_k=20):
  """Performs a local (numpy) calculation of the global average precision.

  Only the top_k predictions are taken for each of the videos.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.
    top_k: How many predictions to use per video.

  Returns:
    float: The global average precision.
  """

  def flatten(l):
      """ Merges a list of lists into a single list. """
      return [item for sublist in l for item in sublist]

  gap_calculator = AveragePrecisionCalculator()
  sparse_predictions, sparse_labels, num_positives = top_k_by_class(predictions, actuals, top_k)
  gap_calculator.accumulate(flatten(sparse_predictions), flatten(sparse_labels), sum(num_positives))
  return gap_calculator.peek_ap_at_n()


def top_k_triplets(predictions, labels, k=20):
  """Get the top_k for a 1-d numpy array. Returns a sparse list of tuples in
  (prediction, class) format"""
  m = len(predictions)
  k = min(k, m)
  indices = np.argpartition(predictions, -k)[-k:]
  return [(index, predictions[index], labels[index]) for index in indices]


class EvaluationMetrics(object):
  """A class to store the evaluation metrics."""

  def __init__(self, num_class, top_k, accumulate_per_tag=False):
    """Construct an EvaluationMetrics object to store the evaluation metrics.

    Args:
      num_class: A positive integer specifying the number of classes.
      top_k: A positive integer specifying how many predictions are considered per video.

    Raises:
      ValueError: An error occurred when MeanAveragePrecisionCalculator cannot
        not be constructed.
    """
    self.sum_hit_at_one = 0.0
    self.sum_perr = 0.0
    self.sum_loss = 0.0
    self.map_calculator = MeanAveragePrecisionCalculator(num_class)
    self.global_ap_calculator = AveragePrecisionCalculator()
    self.pr_calculator = PRCalculator()
    self.pr_calculator_per_tag = PRCalculatorPerTag(num_class)
    self.accumulate_per_tag = accumulate_per_tag

    self.top_k = top_k
    self.num_examples = 0
    self.nums_per_tag = np.zeros(num_class)
    self.tag_corrlation = np.zeros((num_class, num_class))
    self.tag_confidence = np.zeros(num_class)


def top_k_by_class(predictions, labels, k=20):
  """Extracts the top k predictions for each video, sorted by class.

  Args:
    predictions: A numpy matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    k: the top k non-zero entries to preserve in each prediction.

  Returns:
    A tuple (predictions,labels, true_positives). 'predictions' and 'labels'
    are lists of lists of floats. 'true_positives' is a list of scalars. The
    length of the lists are equal to the number of classes. The entries in the
    predictions variable are probability predictions, and
    the corresponding entries in the labels variable are the ground truth for
    those predictions. The entries in 'true_positives' are the number of true
    positives for each class in the ground truth.

  Raises:
    ValueError: An error occurred when the k is not a positive integer.
  """
  if k <= 0:
    raise ValueError("k must be a positive integer.")
  k = min(k, predictions.shape[1])
  num_classes = predictions.shape[1]
  prediction_triplets= []
  for video_index in range(predictions.shape[0]):
    prediction_triplets.extend(top_k_triplets(predictions[video_index],labels[video_index], k))
  out_predictions = [[] for v in range(num_classes)]
  out_labels = [[] for v in range(num_classes)]
  for triplet in prediction_triplets:
    out_predictions[triplet[0]].append(triplet[1])
    out_labels[triplet[0]].append(triplet[2])
  out_true_positives = [np.sum(labels[:,i]) for i in range(num_classes)]

  return out_predictions, out_labels, out_true_positives


class PRCalculator():
  def __init__(self):
      # use only two threshold to save eval time
      self.threshold_dict={0.5:0, 0.1:1} #TODO(jefxiong, range from 0.9~0.01)
      self.precision = np.zeros((len(self.threshold_dict)))
      self.recall = np.zeros((len(self.threshold_dict)))
      self.accumulate_count = np.zeros((len(self.threshold_dict)))

  def accumulate(self, predictions, actuals):
      """
      predictions: n_example X n_classes
      actuals: n_example X n_classes
      """
      #assert isinstance(predictions, np.ndarray)
      #assert isinstance(actuals, np.ndarray)
      n_example = predictions.shape[0]

      precision_all = np.zeros((n_example, len(self.threshold_dict)))
      recall_all = np.zeros((n_example, len(self.threshold_dict)))
      for i in range(n_example):
        gt_index = np.nonzero(actuals[i])[0]
        for th, th_index in self.threshold_dict.items():
          pred_index = np.nonzero(predictions[i]>th)[0]
          tp = np.sum([actuals[i][k] for k in pred_index])
          precision_all[i][th_index]  = tp*1.0/len(pred_index) if len(pred_index)>0 else np.nan
          recall_all[i][th_index]  = tp*1.0/len(gt_index) if len(gt_index)>0 else np.nan


      valid_accumlate = (np.sum(~np.isnan(precision_all), axis=0)) != 0
      self.accumulate_count = self.accumulate_count + valid_accumlate

      precision_all = np.nansum(precision_all,axis=0)/(np.sum(~np.isnan(precision_all), axis=0)+1e-10)
      recall_all = np.nansum(recall_all,axis=0)/(np.sum(~np.isnan(recall_all), axis=0)+1e-10)

      self.precision = precision_all + self.precision
      self.recall = recall_all + self.recall

  def get_precision_at_conf(self, th=0.5):
      index = self.threshold_dict[th]
      return self.precision[index]/(1e-10+self.accumulate_count[index])

  def get_recall_at_conf(self, th=0.5):
      index = self.threshold_dict[th]
      return self.recall[index]/(1e-10+self.accumulate_count[index])

  def clear(self):
      self.accumulate_count = np.zeros((len(self.threshold_dict)))
      self.precision = np.zeros((len(self.threshold_dict)))
      self.recall = np.zeros((len(self.threshold_dict)))


class PRCalculatorPerTag():
  def __init__(self, tag_num):
    self.tag_num = tag_num
    self.pr_calculators = []
    for i in range(self.tag_num):
      self.pr_calculators.append(PRCalculator())

  #@count_func_time
  def accumulate(self, predictions, actuals):
    """
    predictions: n_example X n_classes
    actuals: n_example X n_classes
    """
    #n_example X n_classes ==> n_classes * [n_example x 1]
    pred_per_tag_list = np.expand_dims(predictions.transpose(), -1)
    actuals_per_tag_list = np.expand_dims(actuals.transpose(), -1)

    for i in range(self.tag_num):
      self.pr_calculators[i].accumulate(pred_per_tag_list[i], actuals_per_tag_list[i])
    #ret = list(map(map_func, self.pr_calculators, pred_per_tag_list, actuals_per_tag_list))

  def get_precision_list(self, th=0.5):
    return [self.pr_calculators[i].get_precision_at_conf(th) for i in range(self.tag_num)]

  def get_recall_list(self, th=0.5):
    return [self.pr_calculators[i].get_recall_at_conf(th) for i in range(self.tag_num)]

  def clear(self):
    for i in range(self.tag_num):
      self.pr_calculators[i].clear()


class AveragePrecisionCalculator(object):
  """Calculate the average precision and average precision at n."""

  def __init__(self, top_n=None):
    """Construct an AveragePrecisionCalculator to calculate average precision.

    This class is used to calculate the average precision for a single label.

    Args:
      top_n: A positive Integer specifying the average precision at n, or
        None to use all provided data points.

    Raises:
      ValueError: An error occurred when the top_n is not a positive integer.
    """
    if not ((isinstance(top_n, int) and top_n >= 0) or top_n is None):
      raise ValueError("top_n must be a positive integer or None.")

    self._top_n = top_n  # average precision at n
    self._total_positives = 0  # total number of positives have seen
    self._heap = []  # max heap of (prediction, actual)

  @property
  def heap_size(self):
    """Gets the heap size maintained in the class."""
    return len(self._heap)

  @property
  def num_accumulated_positives(self):
    """Gets the number of positive samples that have been accumulated."""
    return self._total_positives

  def accumulate(self, predictions, actuals, num_positives=None):
    """Accumulate the predictions and their ground truth labels.

    After the function call, we may call peek_ap_at_n to actually calculate
    the average precision.
    Note predictions and actuals must have the same shape.

    Args:
      predictions: a list storing the prediction scores.
      actuals: a list storing the ground truth labels. Any value
      larger than 0 will be treated as positives, otherwise as negatives.
      num_positives = If the 'predictions' and 'actuals' inputs aren't complete,
      then it's possible some true positives were missed in them. In that case,
      you can provide 'num_positives' in order to accurately track recall.

    Raises:
      ValueError: An error occurred when the format of the input is not the
      numpy 1-D array or the shape of predictions and actuals does not match.
    """
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if not num_positives is None:
      if not isinstance(num_positives, numbers.Number) or num_positives < 0:
        raise ValueError("'num_positives' was provided but it wan't a nonzero number.")

    if not num_positives is None:
      self._total_positives += num_positives
    else:
      self._total_positives += np.size(np.where(actuals > 0))
    topk = self._top_n
    heap = self._heap

    for i in range(np.size(predictions)):
      if topk is None or len(heap) < topk:
        heapq.heappush(heap, (predictions[i], actuals[i]))
      else:
        if predictions[i] > heap[0][0]:  # heap[0] is the smallest
          heapq.heappop(heap)
          heapq.heappush(heap, (predictions[i], actuals[i]))

  def clear(self):
    """Clear the accumulated predictions."""
    self._heap = []
    self._total_positives = 0

  def peek_ap_at_n(self):
    """Peek the non-interpolated average precision at n.

    Returns:
      The non-interpolated average precision at n (default 0).
      If n is larger than the length of the ranked list,
      the average precision will be returned.
    """
    if self.heap_size <= 0:
      return 0
    predlists = np.array(list(zip(*self._heap)))

    ap = self.ap_at_n(predlists[0],
                      predlists[1],
                      n=self._top_n,
                      total_num_positives=self._total_positives)
    return ap

  @staticmethod
  def ap(predictions, actuals):
    """Calculate the non-interpolated average precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      actuals: a numpy 1-D array storing the ground truth labels. Any value
      larger than 0 will be treated as positives, otherwise as negatives.

    Returns:
      The non-interpolated average precision at n.
      If n is larger than the length of the ranked list,
      the average precision will be returned.

    Raises:
      ValueError: An error occurred when the format of the input is not the
      numpy 1-D array or the shape of predictions and actuals does not match.
    """
    return AveragePrecisionCalculator.ap_at_n(predictions,
                                              actuals,
                                              n=None)

  @staticmethod
  def ap_at_n(predictions, actuals, n=20, total_num_positives=None):
    """Calculate the non-interpolated average precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      actuals: a numpy 1-D array storing the ground truth labels. Any value
      larger than 0 will be treated as positives, otherwise as negatives.
      n: the top n items to be considered in ap@n.
      total_num_positives : (optionally) you can specify the number of total
        positive
      in the list. If specified, it will be used in calculation.

    Returns:
      The non-interpolated average precision at n.
      If n is larger than the length of the ranked list,
      the average precision will be returned.

    Raises:
      ValueError: An error occurred when
      1) the format of the input is not the numpy 1-D array;
      2) the shape of predictions and actuals does not match;
      3) the input n is not a positive integer.
    """
    if len(predictions) != len(actuals):
      raise ValueError("the shape of predictions and actuals does not match.")

    if n is not None:
      if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be 'None' or a positive integer."
                         " It was '%s'." % n)

    ap = 0.0

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # add a shuffler to avoid overestimating the ap
    predictions, actuals = AveragePrecisionCalculator._shuffle(predictions,
                                                               actuals)
    sortidx = sorted(
        range(len(predictions)),
        key=lambda k: predictions[k],
        reverse=True)

    if total_num_positives is None:
      numpos = np.size(np.where(actuals > 0))
    else:
      numpos = total_num_positives

    if numpos == 0:
      return 0

    if n is not None:
      numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0

    # calculate the ap
    r = len(sortidx)
    if n is not None:
      r = min(r, n)
    for i in range(r):
      if actuals[sortidx[i]] > 0:
        poscount += 1
        ap += poscount / (i + 1) * delta_recall
    return ap

  @staticmethod
  def _shuffle(predictions, actuals):
    random.seed(0)
    suffidx = random.sample(range(len(predictions)), len(predictions))
    predictions = predictions[suffidx]
    actuals = actuals[suffidx]
    return predictions, actuals

  @staticmethod
  def _zero_one_normalize(predictions, epsilon=1e-7):
    """Normalize the predictions to the range between 0.0 and 1.0.

    For some predictions like SVM predictions, we need to normalize them before
    calculate the interpolated average precision. The normalization will not
    change the rank in the original list and thus won't change the average
    precision.

    Args:
      predictions: a numpy 1-D array storing the sparse prediction scores.
      epsilon: a small constant to avoid denominator being zero.

    Returns:
      The normalized prediction.
    """
    denominator = np.max(predictions) - np.min(predictions)
    ret = (predictions - np.min(predictions)) / np.max(denominator,
                                                             epsilon)
    return ret


class MeanAveragePrecisionCalculator(object):
  """This class is to calculate mean average precision.
  """

  def __init__(self, num_class):
    """Construct a calculator to calculate the (macro) average precision.

    Args:
      num_class: A positive Integer specifying the number of classes.
      top_n_array: A list of positive integers specifying the top n for each
      class. The top n in each class will be used to calculate its average
      precision at n.
      The size of the array must be num_class.

    Raises:
      ValueError: An error occurred when num_class is not a positive integer;
      or the top_n_array is not a list of positive integers.
    """
    if not isinstance(num_class, int) or num_class <= 1:
      raise ValueError("num_class must be a positive integer.")

    self._ap_calculators = []  # member of AveragePrecisionCalculator
    self._num_class = num_class  # total number of classes
    for i in range(num_class):
      self._ap_calculators.append(AveragePrecisionCalculator())

  def accumulate(self, predictions, actuals, num_positives=None):
    """Accumulate the predictions and their ground truth labels.

    Args:
      predictions: A list of lists storing the prediction scores. The outer
      dimension corresponds to classes.
      actuals: A list of lists storing the ground truth labels. The dimensions
      should correspond to the predictions input. Any value
      larger than 0 will be treated as positives, otherwise as negatives.
      num_positives: If provided, it is a list of numbers representing the
      number of true positives for each class. If not provided, the number of
      true positives will be inferred from the 'actuals' array.

    Raises:
      ValueError: An error occurred when the shape of predictions and actuals
      does not match.
    """
    if not num_positives:
      num_positives = [None for i in predictions.shape[1]]

    calculators = self._ap_calculators
    for i in range(len(predictions)):
      calculators[i].accumulate(predictions[i], actuals[i], num_positives[i])

  def clear(self):
    for calculator in self._ap_calculators:
      calculator.clear()

  def is_empty(self):
    return ([calculator.heap_size for calculator in self._ap_calculators] ==
            [0 for _ in range(self._num_class)])

  def peek_map_at_n(self):
    """Peek the non-interpolated mean average precision at n.

    Returns:
      An array of non-interpolated average precision at n (default 0) for each
      class.
    """
    aps = [self._ap_calculators[i].peek_ap_at_n()
           for i in range(self._num_class)]
    return aps