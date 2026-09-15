import {
  type LucideIcon,
  type LucideProps,
  ArrowUp as LArrowUp,
  Bolt,
  Check,
  CheckCircle2,
  ChevronDown as LChevronDown,
  ChevronRight as LChevronRight,
  Coins,
  Download,
  ExternalLink,
  Gift,
  Globe,
  Heart,
  History,
  Image,
  ListOrdered,
  Loader2,
  Mail,
  Menu,
  MessageSquare,
  Monitor,
  Paperclip,
  Phone,
  RefreshCw,
  Search,
  Share2,
  Sparkles,
  Split,
  Square,
  SquarePen,
  User,
  Verified,
  X,
  XCircle,
} from "lucide-react";

// 图标统一走 lucide-react（成熟、风格一致、按需 tree-shake）。这里只保留旧名字的薄壳，
// 让历史组件的 `<CloseIcon width={18} height={18} />` 继续能用；新组件直接 import lucide。
// 默认线宽 1.75：比 lucide 默认的 2 细一点，跟发丝线边框的视觉重量一致。
type IconProps = LucideProps;

function wrap(Icon: LucideIcon) {
  return (p: IconProps) => <Icon strokeWidth={1.75} {...p} />;
}

export const ComposeIcon = wrap(SquarePen);
export const MonitorIcon = wrap(Monitor);
export const SearchIcon = wrap(Search);
export const HeartIcon = wrap(Heart);
export const MailIcon = wrap(Mail);
export const ChatIcon = wrap(MessageSquare);
export const HistoryIcon = wrap(History);
export const PhoneIcon = wrap(Phone);
export const UserIcon = wrap(User);
export const GiftIcon = wrap(Gift);
export const ListIcon = wrap(ListOrdered);
export const ShareIcon = wrap(Share2);
export const CoinIcon = wrap(Coins);
export const SparkleIcon = wrap(Sparkles);
export const MenuIcon = wrap(Menu);
export const GlobeIcon = wrap(Globe);
export const PaperclipIcon = wrap(Paperclip);
export const ImageIcon = wrap(Image);
export const BoltIcon = wrap(Bolt);
export const ChevronDown = wrap(LChevronDown);
export const ArrowUp = wrap(LArrowUp);
export const StopIcon = wrap(Square);
export const CheckIcon = wrap(Check);
export const VerifiedIcon = wrap(Verified);
export const CloseIcon = wrap(X);
export const RefreshIcon = wrap(RefreshCw);
export const DownloadIcon = wrap(Download);
export const ExternalLinkIcon = wrap(ExternalLink);
export const SpinnerIcon = wrap(Loader2);
export const CheckCircleIcon = wrap(CheckCircle2);
export const ErrorCircleIcon = wrap(XCircle);
export const ForkIcon = wrap(Split);
export const ChevronRight = wrap(LChevronRight);
