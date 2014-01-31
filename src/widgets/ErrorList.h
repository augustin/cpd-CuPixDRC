#ifndef ERRORLIST_H
#define ERRORLIST_H

#include <QDockWidget>
#include <QIcon>

namespace Ui {
class ErrorList;
}

class ErrorList : public QDockWidget
{
    Q_OBJECT

public:
    explicit ErrorList(QWidget *parent = 0);
    ~ErrorList();

    void setErrors(int *errors);
    void resizeCols();

public slots:
    void updateFilters();

private:
    Ui::ErrorList *ui;
    QIcon error;
    QIcon warning;
    QIcon information;
};

#endif // ERRORLIST_H
