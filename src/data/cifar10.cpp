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

/* read data */
void readCifar10(const char *filename, float *data_ptr, float *label_ptr, int num_images, int offset)
{
    unsigned char temp = 0;

    std::ifstream file (filename, std::ios::binary);
    if (file.is_open())
    {
        for (int n = 0; n < num_images; n++)
        {
            file.read((char*)&temp, sizeof(temp));
            for (int l = 0; l < num_classes; l++)
            {
                if (l == temp)
                {
                    label_ptr[(n+offset)*num_classes+l] = 1;
                }
                else
                {
                    label_ptr[(n+offset)*num_classes+l] = 0;
                }
            }

            for (int c = 0; c < num_image_chls; c++)
            {
                for (int h = 0; h < num_image_rows; h++)
                {
                    for (int w = 0; w < num_image_cols; w++)
                    {
                        file.read((char*)&temp, sizeof(temp));
                        int index = (((n+offset)*num_image_chls + c)*num_image_rows + h)*num_image_cols + w;
                        data_ptr[index] = temp/256.0f;
                    }
                }
            }
        }
    }
}

/* load training and test data */
void load_cifar10()
{
    num_train_images = 50000;
    num_test_images = 10000;
    num_image_chls = 3;
    num_image_rows = 32;
    num_image_cols = 32;
    num_classes = 10;

    train_data_ptr = new float[num_train_images*num_image_chls*num_image_rows*num_image_cols];
    train_label_ptr = new float[num_train_images*num_classes];
    test_data_ptr = new float[num_test_images*num_image_chls*num_image_rows*num_image_cols];
    test_label_ptr = new float[num_test_images*num_classes];

    int offset = 0;
    readCifar10("data/cifar10/cifar-10-batches-bin/data_batch_1.bin", train_data_ptr, train_label_ptr, 10000, offset);
    offset += 10000;
    readCifar10("data/cifar10/cifar-10-batches-bin/data_batch_2.bin", train_data_ptr, train_label_ptr, 10000, offset);
    offset += 10000;
    readCifar10("data/cifar10/cifar-10-batches-bin/data_batch_3.bin", train_data_ptr, train_label_ptr, 10000, offset);
    offset += 10000;
    readCifar10("data/cifar10/cifar-10-batches-bin/data_batch_4.bin", train_data_ptr, train_label_ptr, 10000, offset);
    offset += 10000;
    readCifar10("data/cifar10/cifar-10-batches-bin/data_batch_5.bin", train_data_ptr, train_label_ptr, 10000, offset);
    readCifar10("data/cifar10/cifar-10-batches-bin/test_batch.bin", test_data_ptr, test_label_ptr, 10000, 0);
}
