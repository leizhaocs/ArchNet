/**********************************************************************
 *
 * Copyright Lei Zhao.
 * contact: leizhao0403@gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **********************************************************************/

#include "includes.h"

extern int num_train_images;
extern int num_test_images;
extern int num_image_chls;
extern int num_image_rows;
extern int num_image_cols;
extern int num_classes;
extern float *train_data_ptr;
extern float *train_label_ptr;
extern float *test_data_ptr;
extern float *test_label_ptr;

/* read training data and labels */
void read(float **data_ptr, float **label_ptr)
{
    *data_ptr = new float[num_train_images*num_image_rows*num_image_cols*num_image_chls];
    *label_ptr = new float[num_train_images*num_classes];

    (*label_ptr)[0] = 0; (*label_ptr)[1] = 1;

    int i = 0;

    (*data_ptr)[i++] = 0.1;(*data_ptr)[i++] = -0.2;(*data_ptr)[i++] = 0.3;(*data_ptr)[i++] = -0.4;(*data_ptr)[i++] = 0.5;
    (*data_ptr)[i++] = 0.3;(*data_ptr)[i++] = 0.8;(*data_ptr)[i++] = 0.3;(*data_ptr)[i++] = 0.4;(*data_ptr)[i++] = 0.2;
    (*data_ptr)[i++] = 0.4;(*data_ptr)[i++] = 0.7;(*data_ptr)[i++] = -0.3;(*data_ptr)[i++] = 0.2;(*data_ptr)[i++] = 0;
    (*data_ptr)[i++] = 0.2;(*data_ptr)[i++] = 0.4;(*data_ptr)[i++] = 0.6;(*data_ptr)[i++] = 0;(*data_ptr)[i++] = 0;
    (*data_ptr)[i++] = 0.4;(*data_ptr)[i++] = -0.6;(*data_ptr)[i++] = 0.8;(*data_ptr)[i++] = 0;(*data_ptr)[i++] = -0.2;

    (*data_ptr)[i++] = 0.2;(*data_ptr)[i++] = 0.2;(*data_ptr)[i++] = 0.1;(*data_ptr)[i++] = 0.2;(*data_ptr)[i++] = 0.7;
    (*data_ptr)[i++] = -0.6;(*data_ptr)[i++] = 0.4;(*data_ptr)[i++] = 0.2;(*data_ptr)[i++] = 0.5;(*data_ptr)[i++] = 0.9;
    (*data_ptr)[i++] = 0.8;(*data_ptr)[i++] = -0.6;(*data_ptr)[i++] = 0.3;(*data_ptr)[i++] = 0.3;(*data_ptr)[i++] = 0.4;
    (*data_ptr)[i++] = 0;(*data_ptr)[i++] = 0.8;(*data_ptr)[i++] = -0.2;(*data_ptr)[i++] = 0.3;(*data_ptr)[i++] = -0.3;
    (*data_ptr)[i++] = -0.4;(*data_ptr)[i++] = 0;(*data_ptr)[i++] = 0.8;(*data_ptr)[i++] = -0.2;(*data_ptr)[i++] = 0;
}

/* load training and test data */
void load_debug()
{
    num_train_images = 1;
    num_test_images = 1;
    num_image_chls = 2;
    num_image_rows = 5;
    num_image_cols = 5;
    num_classes = 2;
    read(&train_data_ptr, &train_label_ptr);
    read(&test_data_ptr, &test_label_ptr);
}
