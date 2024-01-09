# List the disks
lsblk -o NAME,HCTL,SIZE,MOUNTPOINT | grep nvme

# Check the devices
ls -l /dev/disk/by-id/

# Choose and format the disk
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/disk/by-id/nvme-Microsoft_NVMe_Direct_Disk_4bdfa079dc0c00000001

# Mount the disk
# mount point
sudo mkdir -p /mnt/disks/training-disk

# mount the disk
sudo mount -o discard,defaults /dev/disk/by-id/nvme-Microsoft_NVMe_Direct_Disk_4bdfa079dc0c00000001 /mnt/disks/training-disk

# Change the owner
sudo chmod a+w /mnt/disks/training-disk

# Configure automatic mounting on VM restart
sudo cp /etc/fstab /etc/fstab.backup

# Use the blkid command to list the UUID for the disk.
sudo blkid /dev/disk/by-id/nvme-Microsoft_NVMe_Direct_Disk_4bdfa079dc0c00000001
# /dev/disk/by-id/nvme-Microsoft_NVMe_Direct_Disk_4bdfa079dc0c00000001: UUID="4d8e68e0-8b34-46d0-a71d-31a5002af1d5" TYPE="ext4"

cat /etc/fstab:
# UUID=4d8e68e0-8b34-46d0-a71d-31a5002af1d5 /mnt/disks/training-disk ext4 discard,defaults 0 2