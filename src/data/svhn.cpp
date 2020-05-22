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
float *readSVHNData(const char *filename, int &number_of_images, int &n_rows, int &n_cols)
{
    std::ifstream file (filename, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));

        float *ptr = new float[number_of_images*n_rows*n_cols];
        for (int i = 0; i < number_of_images; i++)
        {
            for (int r = 0; r < n_rows; r++)
            {
                for (int c = 0; c < n_cols; c++)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    ptr[i*n_rows*n_cols+r*n_cols+c] = temp/256.0f;
                }
            }
        }
        return ptr;
    }
    return NULL;
}

/* read labels */
float *readSVHNLabels(const char *filename, int &number_of_labels)
{
    std::ifstream file (filename, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_labels, sizeof(number_of_labels));

        float *ptr = new float[number_of_labels*10];
        for (int i = 0; i < number_of_labels; i++)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            for (int j = 0; j < 10; j++)
            {
                if (j == temp)
                {
                    ptr[i*10+j] = 1;
                }
                else
                {
                    ptr[i*10+j] = 0;
                }
            }
        }
        return ptr;
    }
    return NULL;
}

/* load training and test data */
void load_svhn()
{
    train_data_ptr = readSVHNData("data/svhn/train_data", num_train_images, num_image_rows, num_image_cols);
    train_label_ptr = readSVHNLabels("data/svhn/train_label", num_train_images);
    test_data_ptr = readSVHNData("data/svhn/test_data", num_test_images, num_image_rows, num_image_cols);
    test_label_ptr = readSVHNLabels("data/svhn/test_label", num_test_images);

    num_image_chls = 1;
    num_classes = 10;
}
