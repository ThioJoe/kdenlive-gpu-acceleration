/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * The module's page in Kdenlive's settings dialog.
 *
 * The module loads this library in Kdenlive's main program and calls
 * gpufx_settings_init() (see src/settings_page_api.h). The page watches for
 * Kdenlive's settings dialog, a KConfigDialog named "settings", and adds
 * itself with KConfigDialog::addPage(). It has its own KCoreConfigSkeleton on
 * the [gpufx] group of Kdenlive's settings file, so the dialog's OK, Apply,
 * Cancel and Defaults buttons handle it like Kdenlive's own pages. When the
 * settings are saved, the module reads them again: the change applies to the
 * monitor at once, and render processes read the file when they start.
 */

#include "settings_page_api.h"

#include <KConfigDialog>
#include <KCoreConfigSkeleton>
#include <KPageWidgetItem>
#include <KSharedConfig>

#include <QApplication>
#include <QCheckBox>
#include <QDir>
#include <QDirIterator>
#include <QEvent>
#include <QFile>
#include <QGridLayout>
#include <QGroupBox>
#include <QHash>
#include <QIcon>
#include <QLabel>
#include <QPointer>
#include <QStandardPaths>
#include <QThread>
#include <QVBoxLayout>
#include <QXmlStreamReader>

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#ifdef GPUFX_SETTINGS_SELFTEST
#include <QAbstractNativeEventFilter>
#include <QAction>
#include <QElapsedTimer>
#include <QMainWindow>
#include <QMetaEnum>
#include <QTimer>
#include <QFileInfo>
#include <QWindow>

#include <atomic>
#include <cstdlib>
#endif

#if defined(_WIN32)
#include <stdlib.h>
#include <windows.h>
#ifdef GPUFX_SETTINGS_SELFTEST
#include <tlhelp32.h>
#endif
#define GPUFX_SETTINGS_EXPORT __declspec(dllexport)
#else
#define GPUFX_SETTINGS_EXPORT __attribute__((visibility("default")))
#endif

namespace {

const gpufx_settings_host *g_host = nullptr;

/** Dynamic property of the dialog: its page item, once the page is added */
const char *const kPageProperty = "gpufx_page";

/** Runs a function in a thread of its own. Not QThread::create(), which
 *  hands Qt a std::future: this library has its own copy of the C++ runtime,
 *  whose objects Kdenlive's Qt must not use (a thread waiting on such a
 *  future hangs in Kdenlive). */
class FunctionThread : public QThread
{
public:
    explicit FunctionThread(std::function<void()> function)
        : m_function(std::move(function))
    {
        connect(this, &QThread::finished, this, &QObject::deleteLater);
    }

protected:
    void run() override
    {
        m_function();
    }

private:
    std::function<void()> m_function;
};

/** Something the module can run on the GPU, and its switch on the page */
struct Service
{
    QString key; // <kind>:<service>, its key in the settings file
    QString kind;
    QString service;
    QString label;
};

/** The names Kdenlive shows for MLT services, from its effect or transition
 *  definitions (several effects can use the same service) */
QHash<QString, QStringList> kdenliveNames(const QString &folder)
{
    QHash<QString, QStringList> names;
    const QStringList dirs = QStandardPaths::locateAll(QStandardPaths::AppDataLocation, folder, QStandardPaths::LocateDirectory);
    for (const QString &dir : dirs) {
        QDirIterator it(dir, {QStringLiteral("*.xml")}, QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext()) {
            QFile file(it.next());
            if (!file.open(QIODevice::ReadOnly)) {
                continue;
            }
            QXmlStreamReader xml(&file);
            QString tag;
            while (!xml.atEnd() && xml.readNext() != QXmlStreamReader::Invalid) {
                if (!xml.isStartElement()) {
                    continue;
                }
                if (xml.name() == QLatin1String("effect") || xml.name() == QLatin1String("transition")) {
                    tag = xml.attributes().value(QLatin1String("tag")).toString();
                } else if (xml.name() == QLatin1String("name") && !tag.isEmpty()) {
                    const QString name = xml.readElementText().trimmed();
                    if (!name.isEmpty() && !names[tag].contains(name)) {
                        names[tag].append(name);
                    }
                    break;
                }
            }
        }
    }
    for (QStringList &list : names) {
        list.sort(Qt::CaseInsensitive);
    }
    return names;
}

/** Kdenlive's names, for when its definitions are not found */
QString fallbackName(const QString &key)
{
    static const QHash<QString, QString> names = {
        {QStringLiteral("effect:dropshadow"), QStringLiteral("Drop Shadow")},
        {QStringLiteral("effect:box_blur"), QStringLiteral("BoxBlur")},
        {QStringLiteral("effect:boxblur"), QStringLiteral("Box Blur")},
        {QStringLiteral("effect:brightness"), QStringLiteral("Fade in, Fade out, Intensity (Brightness)")},
        {QStringLiteral("effect:lift_gamma_gain"), QStringLiteral("Lift/gamma/gain")},
        {QStringLiteral("effect:frei0r.squareblur"), QStringLiteral("Square Blur")},
        {QStringLiteral("effect:frei0r.glow"), QStringLiteral("Glow")},
        {QStringLiteral("effect:frei0r.softglow"), QStringLiteral("Soft Glow")},
        {QStringLiteral("effect:frei0r.sharpness"), QStringLiteral("Sharpen")},
        {QStringLiteral("effect:qtblend"), QStringLiteral("Transform")},
        {QStringLiteral("transition:qtblend"), QStringLiteral("Composite and transform")},
    };
    return names.value(key);
}

int kindOrder(const QString &kind)
{
    if (kind == QLatin1String("effect")) {
        return 0;
    }
    return kind == QLatin1String("transition") ? 1 : 2;
}

/** The services of the module, effects first, sorted by name */
const std::vector<Service> &services()
{
    static const std::vector<Service> result = [] {
        const QHash<QString, QStringList> effects = kdenliveNames(QStringLiteral("effects"));
        const QHash<QString, QStringList> transitions = kdenliveNames(QStringLiteral("transitions"));
        std::vector<Service> list;
        for (const QString &key : QString::fromUtf8(g_host->services).split(QLatin1Char(','), Qt::SkipEmptyParts)) {
            Service service;
            service.key = key;
            service.kind = key.section(QLatin1Char(':'), 0, 0);
            service.service = key.section(QLatin1Char(':'), 1);
            if (service.kind == QLatin1String("conversion")) {
                if (service.service == QLatin1String("avformat")) {
                    service.label = QStringLiteral("Video decoding to RGBA");
                } else if (service.service == QLatin1String("avcolor_space")) {
                    service.label = QStringLiteral("Output conversion to YUV");
                }
            } else {
                const QStringList names = (service.kind == QLatin1String("transition") ? transitions : effects).value(service.service);
                service.label = names.isEmpty() ? fallbackName(key) : names.join(QStringLiteral(", "));
                if (key == QLatin1String("transition:qtblend")) {
                    service.label = QStringLiteral("Track compositing (%1)").arg(service.label);
                }
            }
            if (service.label.isEmpty()) {
                service.label = service.service;
            }
            list.push_back(service);
        }
        std::stable_sort(list.begin(), list.end(), [](const Service &a, const Service &b) {
            if (kindOrder(a.kind) != kindOrder(b.kind)) {
                return kindOrder(a.kind) < kindOrder(b.kind);
            }
            return a.kind == QLatin1String("effect") && a.label.compare(b.label, Qt::CaseInsensitive) < 0;
        });
        return list;
    }();
    return result;
}

/** The settings of the page, in the group the module reads */
class Settings : public KCoreConfigSkeleton
{
public:
    explicit Settings(QObject *parent)
        : KCoreConfigSkeleton(KSharedConfig::openConfig(), parent)
        , m_values(new bool[services().size()])
    {
        setCurrentGroup(QString::fromUtf8(g_host->group));
        addItemBool(QStringLiteral("enabled"), m_enabled, true);
        for (size_t i = 0; i < services().size(); ++i) {
            addItemBool(services()[i].key, m_values[i], true);
        }
        read();
    }

    bool enabled() const
    {
        return m_enabled;
    }

    bool value(size_t service) const
    {
        return m_values[service];
    }

private:
    bool m_enabled = true;
    std::unique_ptr<bool[]> m_values;
};

/** Text for the description of the GPU the module gives */
QString gpuStatus(const QString &description)
{
    QHash<QString, QString> values;
    for (const QString &line : description.split(QLatin1Char('\n'), Qt::SkipEmptyParts)) {
        values.insert(line.section(QLatin1Char('='), 0, 0), line.section(QLatin1Char('='), 1));
    }
    if (values.value(QStringLiteral("available")) != QLatin1String("1")) {
        return QStringLiteral("No usable GPU: %1. Everything runs on the CPU.").arg(values.value(QStringLiteral("status")));
    }
    return QStringLiteral("GPU: %1 (compute capability %2, CUDA driver %3, %4 MB)")
        .arg(values.value(QStringLiteral("device")), values.value(QStringLiteral("compute_capability")), values.value(QStringLiteral("cuda_driver")),
             values.value(QStringLiteral("memory_mb")));
}

/** Fills the label with the GPU's description, which can take a moment the
 *  first time CUDA is initialized */
void showGpuStatus(QLabel *label)
{
    QPointer<QLabel> target(label);
    auto *thread = new FunctionThread([target]() {
        char text[2048] = {};
        g_host->describe_gpu(text, sizeof(text));
        const QString status = gpuStatus(QString::fromUtf8(text));
        QMetaObject::invokeMethod(
            QCoreApplication::instance(),
            [target, status]() {
                if (target) {
                    target->setText(status);
                }
            },
            Qt::QueuedConnection);
    });
    thread->start();
}

QString pageIcon()
{
    for (const char *name : {"gpu", "cpu", "preferences-system-performance", "video-display"}) {
        if (QIcon::hasThemeIcon(QLatin1String(name))) {
            return QLatin1String(name);
        }
    }
    return QStringLiteral("configure");
}

#ifdef GPUFX_SETTINGS_SELFTEST
/*
 * A variant of the page for the workflow's check in Kdenlive itself, which
 * is not installed: once Kdenlive's main window is up, it opens the settings
 * dialog, shows the page, saves a screenshot of the dialog and a report in
 * the folder GPUFX_SELFTEST_OUTPUT names, and ends Kdenlive. A report also
 * says where it stopped when something does not happen in time. With
 * GPUFX_SELFTEST_PAGE=0, it opens the dialog without adding the page, to
 * compare with Kdenlive on its own.
 */
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

bool selftestWantsPage()
{
    return qgetenv("GPUFX_SELFTEST_PAGE") != "0";
}

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
                    // Not from this function: Qt does not run a timer again
                    // while its function runs, which would stop the heartbeat
                    // if opening the dialog runs an event loop
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
#endif

void addPage(KConfigDialog *dialog)
{
    auto *settings = new Settings(dialog);
    auto *page = new QWidget;
    page->setObjectName(QStringLiteral("gpufx_page"));
    auto *layout = new QVBoxLayout(page);

    // The widgets named kcfg_<key> are handled by the dialog
    auto *enabled = new QCheckBox(QStringLiteral("Use the GPU for the effects and conversions below"), page);
    enabled->setObjectName(QStringLiteral("kcfg_enabled"));
    enabled->setChecked(settings->enabled());
    layout->addWidget(enabled);
    auto *about = new QLabel(QStringLiteral("They run on the NVIDIA GPU when possible, with the same result as on the CPU, and anything the GPU "
                                            "cannot do stays on the CPU. Changes apply to the monitor right away and to renders started afterwards."),
                             page);
    about->setWordWrap(true);
    layout->addWidget(about);
    auto *status = new QLabel(QStringLiteral("Checking the GPU…"), page);
    status->setObjectName(QStringLiteral("gpufx_status"));
    status->setWordWrap(true);
    status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(status);

    // The effects in two columns, read down
    auto *effects = new QGroupBox(QStringLiteral("Effects"), page);
    auto *effectsLayout = new QGridLayout(effects);
    const int effectCount = int(std::count_if(services().begin(), services().end(), [](const Service &service) {
        return service.kind == QLatin1String("effect");
    }));
    const int rows = (effectCount + 1) / 2;
    auto *others = new QGroupBox(QStringLiteral("Compositing and conversions"), page);
    auto *othersLayout = new QVBoxLayout(others);
    int effectIndex = 0;
    for (size_t i = 0; i < services().size(); ++i) {
        const Service &service = services()[i];
        const bool effect = service.kind == QLatin1String("effect");
        auto *box = new QCheckBox(service.label, effect ? effects : others);
        box->setObjectName(QStringLiteral("kcfg_") + service.key);
        box->setToolTip(QStringLiteral("MLT %1: %2").arg(service.kind == QLatin1String("conversion") ? QStringLiteral("service") : service.kind, service.service));
        box->setChecked(settings->value(i));
        if (effect) {
            effectsLayout->addWidget(box, effectIndex % rows, effectIndex / rows);
            ++effectIndex;
        } else {
            othersLayout->addWidget(box);
        }
    }
    layout->addWidget(effects);
    layout->addWidget(others);
    layout->addStretch();
    // The switches of the services only matter with GPU processing on
    effects->setEnabled(settings->enabled());
    others->setEnabled(settings->enabled());
    QObject::connect(enabled, &QCheckBox::toggled, effects, &QWidget::setEnabled);
    QObject::connect(enabled, &QCheckBox::toggled, others, &QWidget::setEnabled);

    KPageWidgetItem *item = dialog->addPage(page, settings, QStringLiteral("GPU Acceleration"), pageIcon(), QStringLiteral("GPU acceleration with NVIDIA CUDA"));
    dialog->setProperty(kPageProperty, QVariant::fromValue<QObject *>(item));
    QObject::connect(settings, &KCoreConfigSkeleton::configChanged, settings, []() {
        g_host->reload_settings();
    });
    showGpuStatus(status);
#ifdef GPUFX_SETTINGS_SELFTEST
    selftestReport(dialog, item, status);
#endif
}

/** Adds the page to Kdenlive's settings dialog when it opens */
class PageInstaller : public QObject
{
public:
    using QObject::QObject;

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        // Before the dialog's own showEvent(), which loads the pages' values
        if (event->type() == QEvent::Show && watched->isWidgetType()) {
            auto *dialog = qobject_cast<KConfigDialog *>(watched);
            if (dialog && dialog->objectName() == QLatin1String("settings") && !dialog->property(kPageProperty).isValid()) {
#ifdef GPUFX_SETTINGS_SELFTEST
                if (!selftestWantsPage()) {
                    dialog->setProperty(kPageProperty, true);
                    selftestReport(dialog, nullptr, nullptr);
                    return QObject::eventFilter(watched, event);
                }
#endif
                addPage(dialog);
            }
        }
        return QObject::eventFilter(watched, event);
    }
};

/** Render processes started by Kdenlive inherit the path of its settings
 *  file, which depends on how Kdenlive was installed or started */
void exportSettingsFile()
{
    QString file = KSharedConfig::openConfig()->name();
    if (QDir::isRelativePath(file)) {
        file = QStandardPaths::writableLocation(QStandardPaths::GenericConfigLocation) + QLatin1Char('/') + file;
    }
    file = QDir::toNativeSeparators(file);
#if defined(_WIN32)
    const auto *path = reinterpret_cast<const wchar_t *>(file.utf16());
    _wputenv_s(L"GPUFX_CONFIG_FILE", path);
    SetEnvironmentVariableW(L"GPUFX_CONFIG_FILE", path);
#else
    qputenv("GPUFX_CONFIG_FILE", QFile::encodeName(file));
#endif
    g_host->reload_settings();
}

void start()
{
    QCoreApplication *app = QCoreApplication::instance();
    if (!qobject_cast<QApplication *>(app)) {
        return;
    }
    exportSettingsFile();
#ifdef GPUFX_SETTINGS_SELFTEST
    selftestStart();
#endif
    // The filter must live in the application's thread
    auto *installer = new PageInstaller;
    installer->moveToThread(app->thread());
    QMetaObject::invokeMethod(
        app,
        [installer, app]() {
            installer->setParent(app);
            app->installEventFilter(installer);
        },
        Qt::AutoConnection);
}

/** Called while the application is being constructed: starts once it is */
void startLater()
{
    QMetaObject::invokeMethod(QCoreApplication::instance(), &start, Qt::QueuedConnection);
}

} // namespace

extern "C" GPUFX_SETTINGS_EXPORT int gpufx_settings_init(const gpufx_settings_host *host)
{
    if (g_host) {
        return 1;
    }
    if (!host || host->version < GPUFX_SETTINGS_API_VERSION) {
        return 0;
    }
    g_host = host;
    if (QCoreApplication::instance()) {
        start();
    } else {
        // Kdenlive creates its application before loading MLT, others may not
        qAddPreRoutine(startLater);
    }
    return 1;
}
