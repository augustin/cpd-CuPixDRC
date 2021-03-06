#ifndef DRCWIND_H
#define DRCWIND_H

#include <QMainWindow>

namespace Ui {
class DRCwind;
}
class Chip;

class DRCwind : public QMainWindow
{
    Q_OBJECT

public:
    explicit DRCwind(QWidget *parent = 0);
    ~DRCwind();

private slots:
    void on_actionOpen_triggered();
    void on_actionRunDRC_triggered();

    void on_actionTestcase_triggered();

private:
    Ui::DRCwind *ui;
    Chip* chip;

    QByteArray data;
    int imgW;
    int imgH;
#ifdef CUDA
    int cudaDevice;
#endif
};

#endif // DRCWIND_H
