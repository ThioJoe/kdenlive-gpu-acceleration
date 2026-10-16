/*
    SPDX-FileCopyrightText: 2026 ThioJoe
    SPDX-License-Identifier: GPL-3.0-only
*/

/*
 * A variant of the page for the workflow's check in Kdenlive itself, which
 * is not installed: once Kdenlive's main window is up, it opens the settings
 * dialog, shows the page, saves a screenshot of the dialog and a report in
 * the folder GPUFX_SELFTEST_OUTPUT names, and ends Kdenlive. A report also
 * says where it stopped when something does not happen in time. With
 * GPUFX_SELFTEST_PAGE=0, it opens the dialog without adding the page, to
 * compare with Kdenlive on its own.
 */

#include "selftest.h"

#include "functionthread.h"

#include <KConfigDialog>
#include <KPageWidgetItem>

#include <QAbstractNativeEventFilter>
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QIcon>
#include <QLabel>
#include <QMainWindow>
#include <QMetaEnum>
#include <QPixmap>
#include <QTimer>
#include <QWindow>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace {

QElapsedTimer g_selftestClock;
/** What the check is at, for the reports */
std::atomic<int> g_selftestStage{0};
const char *const kSelftestStages[] = {"waiting for the main window", "opening the settings dialog", "waiting for the dialog",
                                       "showing the dialog"};
/** The last time the main thread ran the check's timer */
std::atomic<qint64> g_selftestHeartbeat{0};
/** Whether Kdenlive's settings dialog was shown */
bool g_selftestDialogShown = false;
/** How many widgets the application had when the dialog was shown */
std::atomic<int> g_selftestWidgets{0};

/** The main thread's events by type and its Windows messages by number, the
 *  last counter taking the larger numbers */
constexpr int kSelftestCounters = 0x500;
std::atomic<quint32> g_selftestEvents[kSelftestCounters];
std::atomic<quint32> g_selftestMessages[kSelftestCounters];
using SelftestCounts = std::vector<quint32>;

/** The last activation events of windows */
struct SelftestActivation
{
    std::atomic<qint64> time{0};
    std::atomic<int> type{0};
    std::atomic<const char *> window{nullptr};
};
constexpr unsigned kSelftestActivations = 32;
SelftestActivation g_selftestActivations[kSelftestActivations];
std::atomic<unsigned> g_selftestActivationCount{0};

/** Counts the main thread's events, for a report when it is stuck */
class SelftestEvents : public QObject
{
public:
    using QObject::QObject;

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        const int type = event->type();
        g_selftestEvents[std::min(type, kSelftestCounters - 1)].fetch_add(1, std::memory_order_relaxed);
        if ((type == QEvent::WindowActivate || type == QEvent::WindowDeactivate || type == QEvent::ActivationChange) && watched->isWidgetType()
            && static_cast<QWidget *>(watched)->isWindow()) {
            SelftestActivation &entry = g_selftestActivations[g_selftestActivationCount.fetch_add(1) % kSelftestActivations];
            entry.time = g_selftestClock.elapsed();
            entry.type = type;
            entry.window = watched->metaObject()->className();
        }
        return QObject::eventFilter(watched, event);
    }
};

#if defined(_WIN32)
/** Counts the main thread's Windows messages */
class SelftestMessages : public QAbstractNativeEventFilter
{
public:
    bool nativeEventFilter(const QByteArray &, void *message, qintptr *) override
    {
        const UINT id = static_cast<const MSG *>(message)->message;
        g_selftestMessages[std::min<UINT>(id, kSelftestCounters - 1)].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
};
#endif

SelftestCounts selftestSnapshot(const std::atomic<quint32> *counters)
{
    SelftestCounts counts(kSelftestCounters);
    for (int i = 0; i < kSelftestCounters; ++i) {
        counts[i] = counters[i].load(std::memory_order_relaxed);
    }
    return counts;
}

QString selftestEventName(int type)
{
    if (type == kSelftestCounters - 1) {
        return QStringLiteral("others");
    }
    const char *name = QMetaEnum::fromType<QEvent::Type>().valueToKey(type);
    return name ? QString::fromLatin1(name) : QString::number(type);
}

#if defined(_WIN32)
QString selftestMessageName(int id)
{
    static const QHash<int, QString> names = {
        {0x0003, QStringLiteral("WM_MOVE")},
        {0x0005, QStringLiteral("WM_SIZE")},
        {0x0006, QStringLiteral("WM_ACTIVATE")},
        {0x0007, QStringLiteral("WM_SETFOCUS")},
        {0x0008, QStringLiteral("WM_KILLFOCUS")},
        {0x000F, QStringLiteral("WM_PAINT")},
        {0x0014, QStringLiteral("WM_ERASEBKGND")},
        {0x0018, QStringLiteral("WM_SHOWWINDOW")},
        {0x001C, QStringLiteral("WM_ACTIVATEAPP")},
        {0x0020, QStringLiteral("WM_SETCURSOR")},
        {0x0021, QStringLiteral("WM_MOUSEACTIVATE")},
        {0x0024, QStringLiteral("WM_GETMINMAXINFO")},
        {0x0046, QStringLiteral("WM_WINDOWPOSCHANGING")},
        {0x0047, QStringLiteral("WM_WINDOWPOSCHANGED")},
        {0x007F, QStringLiteral("WM_GETICON")},
        {0x0083, QStringLiteral("WM_NCCALCSIZE")},
        {0x0084, QStringLiteral("WM_NCHITTEST")},
        {0x0085, QStringLiteral("WM_NCPAINT")},
        {0x0086, QStringLiteral("WM_NCACTIVATE")},
        {0x0113, QStringLiteral("WM_TIMER")},
        {0x0200, QStringLiteral("WM_MOUSEMOVE")},
        {0x0281, QStringLiteral("WM_IME_SETCONTEXT")},
        {0x0282, QStringLiteral("WM_IME_NOTIFY")},
        {0x0400, QStringLiteral("Qt socket notifiers")},
        {0x0401, QStringLiteral("Qt posted events")},
        {0x0402, QStringLiteral("Qt activate notifiers")},
    };
    if (id == kSelftestCounters - 1) {
        return QStringLiteral("others");
    }
    const QString name = names.value(id);
    return QStringLiteral("0x%1").arg(id, 4, 16, QLatin1Char('0')) + (name.isEmpty() ? QString() : QStringLiteral(" ") + name);
}
#endif

/** The counters that grew the most since the snapshot */
QString selftestTop(const std::atomic<quint32> *counters, const SelftestCounts &before, QString (*name)(int))
{
    std::vector<std::pair<quint32, int>> counts;
    for (int i = 0; i < kSelftestCounters; ++i) {
        const quint32 count = counters[i].load(std::memory_order_relaxed) - before[i];
        if (count) {
            counts.emplace_back(count, i);
        }
    }
    std::sort(counts.begin(), counts.end(), std::greater<>());
    QString text;
    for (size_t i = 0; i < counts.size() && i < 12; ++i) {
        text += QStringLiteral("  %1: %2\n").arg(name(counts[i].second)).arg(counts[i].first);
    }
    return text.isEmpty() ? QStringLiteral("  none\n") : text;
}

void selftestFinish(const QString &report, int code)
{
    QFile file(QString::fromLocal8Bit(qgetenv("GPUFX_SELFTEST_OUTPUT")) + QStringLiteral("/kdenlive-report.txt"));
    if (file.open(QIODevice::WriteOnly)) {
        file.write(report.toUtf8());
        file.write(QStringLiteral("after %1 s, %2\n").arg(g_selftestClock.elapsed() / 1000).arg(QLatin1String(kSelftestStages[g_selftestStage])).toUtf8());
        file.close();
    }
    std::_Exit(code);
}

#if defined(_WIN32)
/** The main thread, whose stack a report shows when it is stuck */
DWORD g_selftestMainThread = 0;

/** Module, nearest exported function and offset of a code address */
QString selftestAddress(DWORD64 address)
{
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(address),
                            &module)) {
        return QStringLiteral("0x%1").arg(address, 0, 16);
    }
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(module, path, MAX_PATH);
    const QString name = QFileInfo(QString::fromWCharArray(path)).fileName();
    const auto *base = reinterpret_cast<const BYTE *>(module);
    const DWORD offset = DWORD(address - reinterpret_cast<DWORD64>(base));
    const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
    const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY &directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    DWORD best = 0;
    const char *bestName = nullptr;
    if (directory.Size) {
        const auto *exports = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY *>(base + directory.VirtualAddress);
        const auto *names = reinterpret_cast<const DWORD *>(base + exports->AddressOfNames);
        const auto *ordinals = reinterpret_cast<const WORD *>(base + exports->AddressOfNameOrdinals);
        const auto *functions = reinterpret_cast<const DWORD *>(base + exports->AddressOfFunctions);
        for (DWORD i = 0; i < exports->NumberOfNames; ++i) {
            const DWORD rva = functions[ordinals[i]];
            if (rva <= offset && rva > best) {
                best = rva;
                bestName = reinterpret_cast<const char *>(base + names[i]);
            }
        }
    }
    if (bestName) {
        return QStringLiteral("%1!%2+0x%3").arg(name, QLatin1String(bestName)).arg(offset - best, 0, 16);
    }
    return QStringLiteral("%1+0x%2").arg(name).arg(offset, 0, 16);
}

/** The stack of a thread, unwound with the functions' unwind data */
QString selftestStack(DWORD threadId, int depth)
{
    HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, threadId);
    if (!thread) {
        return QStringLiteral("cannot open thread %1\n").arg(threadId);
    }
    // No allocation while the thread is suspended: it may hold the heap's lock
    DWORD64 frames[64];
    int count = 0;
    depth = std::min(depth, 64);
    SuspendThread(thread);
    CONTEXT context = {};
    context.ContextFlags = CONTEXT_FULL;
    if (GetThreadContext(thread, &context)) {
        while (count < depth && context.Rip) {
            frames[count++] = context.Rip;
            DWORD64 imageBase = 0;
            PRUNTIME_FUNCTION function = RtlLookupFunctionEntry(context.Rip, &imageBase, nullptr);
            if (function) {
                PVOID handlerData = nullptr;
                DWORD64 establisherFrame = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, context.Rip, function, &context, &handlerData, &establisherFrame, nullptr);
            } else {
                // A leaf function: the return address is on top of the stack
                context.Rip = *reinterpret_cast<DWORD64 *>(context.Rsp);
                context.Rsp += 8;
            }
        }
    }
    ResumeThread(thread);
    CloseHandle(thread);
    QString text = QStringLiteral("thread %1%2:\n").arg(threadId).arg(threadId == g_selftestMainThread ? QStringLiteral(" (main)") : QString());
    for (int i = 0; i < count; ++i) {
        text += QStringLiteral("  %1\n").arg(selftestAddress(frames[i]));
    }
    return text;
}

/** The stacks of the other threads of the process */
QString selftestOtherStacks()
{
    QString text;
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return text;
    }
    THREADENTRY32 entry = {};
    entry.dwSize = sizeof(entry);
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID == GetCurrentProcessId() && entry.th32ThreadID != g_selftestMainThread
            && entry.th32ThreadID != GetCurrentThreadId()) {
            text += selftestStack(entry.th32ThreadID, 24);
        }
    }
    CloseHandle(snapshot);
    return text;
}

/** The processor time the main thread used in user and in kernel mode, in ms */
std::pair<qint64, qint64> selftestMainThreadTimes()
{
    std::pair<qint64, qint64> times(-1, -1);
    HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, g_selftestMainThread);
    if (!thread) {
        return times;
    }
    FILETIME creation, exited, kernel, user;
    if (GetThreadTimes(thread, &creation, &exited, &kernel, &user)) {
        const auto ms = [](const FILETIME &time) {
            return qint64((quint64(time.dwHighDateTime) << 32 | time.dwLowDateTime) / 10000);
        };
        times = {ms(user), ms(kernel)};
    }
    CloseHandle(thread);
    return times;
}
#endif

/** What the main thread does, for a report when it does not run the check's
 *  timer: its events and messages since the snapshots, whether it uses the
 *  processor, and its stack */
QString selftestDiagnostics(const SelftestCounts &events, const SelftestCounts &messages)
{
    QString text = QStringLiteral("widgets when the dialog was shown: %1\n").arg(g_selftestWidgets.load());
    text += QStringLiteral("events of the main thread in the last 2 s:\n") + selftestTop(g_selftestEvents, events, selftestEventName);
#if defined(_WIN32)
    text += QStringLiteral("Windows messages of the main thread in the last 2 s:\n") + selftestTop(g_selftestMessages, messages, selftestMessageName);
#else
    Q_UNUSED(messages)
#endif
    const unsigned count = g_selftestActivationCount.load();
    text += QStringLiteral("activation events of windows: %1, the last ones:\n").arg(count);
    for (unsigned i = count > kSelftestActivations ? count - kSelftestActivations : 0; i < count; ++i) {
        const SelftestActivation &entry = g_selftestActivations[i % kSelftestActivations];
        const char *window = entry.window.load();
        text += QStringLiteral("  %1 s %2 %3\n")
                    .arg(entry.time.load() / 1000.0, 0, 'f', 1)
                    .arg(selftestEventName(entry.type.load()), QLatin1String(window ? window : "?"));
    }
#if defined(_WIN32)
    const std::pair<qint64, qint64> before = selftestMainThreadTimes();
    QThread::msleep(1000);
    const std::pair<qint64, qint64> after = selftestMainThreadTimes();
    text += QStringLiteral("processor time of the main thread in 1 s: %1 ms user, %2 ms kernel\n")
                .arg(after.first - before.first)
                .arg(after.second - before.second);
    // A few samples of the main thread, then the other threads
    for (int i = 0; i < 3; ++i) {
        if (i) {
            QThread::msleep(300);
        }
        text += selftestStack(g_selftestMainThread, 64);
    }
    text += selftestOtherStacks();
#endif
    return text;
}

/** The application's windows, for a report when the settings dialog does
 *  not open */
QString selftestWindows()
{
    QString text;
    for (QWidget *widget : QApplication::topLevelWidgets()) {
        if (!widget->isVisible() && !qobject_cast<QMainWindow *>(widget)) {
            continue;
        }
        text += QStringLiteral("%1 \"%2\" \"%3\"%4\n")
                    .arg(QLatin1String(widget->metaObject()->className()), widget->objectName(), widget->windowTitle(),
                         widget->isVisible() ? QString() : QStringLiteral(" (hidden)"));
        for (QLabel *label : widget->findChildren<QLabel *>()) {
            if (label->isVisible() && !label->text().isEmpty()) {
                text += QStringLiteral("  %1\n").arg(label->text().left(300));
            }
        }
    }
    if (QWidget *modal = QApplication::activeModalWidget()) {
        text += QStringLiteral("modal: %1 \"%2\"\n").arg(QLatin1String(modal->metaObject()->className()), modal->windowTitle());
    }
    // Qt Quick windows, like Kdenlive's splash screen
    for (QWindow *window : QGuiApplication::topLevelWindows()) {
        if (window->isVisible()) {
            text += QStringLiteral("window: %1 \"%2\"\n").arg(QLatin1String(window->metaObject()->className()), window->title());
        }
    }
    return text;
}

/** An action of the application. KXmlGui's actions belong to their action
 *  collection, not to the window: they are found in the menus they are in. */
QAction *selftestAction(const QString &name)
{
    for (QWidget *widget : QApplication::allWidgets()) {
        for (QAction *action : widget->actions()) {
            if (action->objectName() == name) {
                return action;
            }
        }
    }
    return nullptr;
}

} // namespace

bool selftestWantsPage()
{
    return qgetenv("GPUFX_SELFTEST_PAGE") != "0";
}

void selftestStart()
{
    g_selftestClock.start();
    QCoreApplication *app = QCoreApplication::instance();
#if defined(_WIN32)
    g_selftestMainThread = GetCurrentThreadId();
    app->installNativeEventFilter(new SelftestMessages);
#endif
    app->installEventFilter(new SelftestEvents(app));
    // From another thread: reports when the main thread does not run the
    // check's timer, or when the whole check takes too long
    auto *watchdog = new FunctionThread([]() {
        for (;;) {
            const SelftestCounts events = selftestSnapshot(g_selftestEvents);
            const SelftestCounts messages = selftestSnapshot(g_selftestMessages);
            QThread::msleep(2000);
            const qint64 now = g_selftestClock.elapsed();
            if (now - g_selftestHeartbeat > 60000) {
                selftestFinish(QStringLiteral("Kdenlive's main thread has not run the check's timer for 60 s\n") + selftestDiagnostics(events, messages), 5);
            }
            if (now > 200000) {
                selftestFinish(QStringLiteral("The check took too long\n") + selftestDiagnostics(events, messages), 6);
            }
        }
    });
    watchdog->start();

    auto *timer = new QTimer(app);
    QObject::connect(timer, &QTimer::timeout, timer, []() {
        g_selftestHeartbeat = g_selftestClock.elapsed();
        if (g_selftestStage == 0) {
            for (QWidget *widget : QApplication::topLevelWidgets()) {
                if (!qobject_cast<QMainWindow *>(widget) || !widget->isVisible()) {
                    continue;
                }
                if (QAction *action = selftestAction(QStringLiteral("options_configure"))) {
                    g_selftestStage = 1;
                    // From a queued call: opened from this timer's function,
                    // the dialog kept Kdenlive's main thread busy with window
                    // activation events for minutes on the Windows runner
                    QMetaObject::invokeMethod(
                        action,
                        [action]() {
                            action->trigger();
                            g_selftestStage = std::max(int(g_selftestStage), 2);
                        },
                        Qt::QueuedConnection);
                    return;
                }
            }
            if (g_selftestClock.elapsed() > 120000) {
                selftestFinish(QStringLiteral("Kdenlive's main window did not appear. Windows:\n") + selftestWindows(), 2);
            }
        } else if (!g_selftestDialogShown && g_selftestClock.elapsed() > 150000) {
            selftestFinish(QStringLiteral("Kdenlive's settings dialog did not open. Windows:\n") + selftestWindows(), 7);
        }
    });
    timer->start(500);
}

/** Shows the page, or the dialog without it, then saves the report and the
 *  screenshot */
void selftestReport(KConfigDialog *dialog, KPageWidgetItem *item, QLabel *status)
{
    g_selftestDialogShown = true;
    g_selftestWidgets = int(QApplication::allWidgets().size());
    g_selftestStage = 3;
    auto *timer = new QTimer(dialog);
    auto clock = std::make_shared<QElapsedTimer>();
    clock->start();
    QObject::connect(timer, &QTimer::timeout, dialog, [=]() {
        if (status && status->text().startsWith(QLatin1String("Checking")) && clock->elapsed() < 60000) {
            return;
        }
        timer->stop();
        if (item) {
            dialog->setCurrentPage(item);
        }
        QTimer::singleShot(1000, dialog, [=]() {
            QString report = QStringLiteral("dialog: %1 \"%2\"\n").arg(QLatin1String(dialog->metaObject()->className()), dialog->objectName());
            report += QStringLiteral("current page: %1\n").arg(dialog->currentPage() ? dialog->currentPage()->name() : QString());
            if (item) {
                report += QStringLiteral("icon: %1\n").arg(item->icon().name());
                report += QStringLiteral("status: %1\n").arg(status->text());
                for (QCheckBox *box : item->widget()->findChildren<QCheckBox *>()) {
                    report += QStringLiteral("%1: %2%3\n").arg(box->objectName(), box->text(), box->isChecked() ? QString() : QStringLiteral(" (off)"));
                }
            } else {
                report += QStringLiteral("without the page (GPUFX_SELFTEST_PAGE=0)\n");
            }
            const bool saved = dialog->grab().save(QString::fromLocal8Bit(qgetenv("GPUFX_SELFTEST_OUTPUT")) + QStringLiteral("/kdenlive-settings.png"));
            selftestFinish(report, saved ? 0 : 3);
        });
    });
    timer->start(200);
}
