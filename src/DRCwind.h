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

private:
    Ui::DRCwind *ui;
    Chip* chip;
};

#endif // DRCWIND_H
