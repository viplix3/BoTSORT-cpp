#include "DataType.h"

void linear_assignment(CostMatrix &cost_matrix, AssociationData &association_data, float cost_threshold = 0.0f);

CostMatrix calc_iou(DetMatrix &bboxe_list_a, DetMatrix &bbox_list_b);